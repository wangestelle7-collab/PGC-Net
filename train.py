from datetime import datetime
from models.PGC import Generator, Generator_Ct_Feature, Discriminator  # Replace with actual models
import random
from torchvision import transforms
from PIL import Image
from Util.vggloss import *  # Replace with actual loss functions
from datasets import ImageDatasetV2  # Ensure correct dataset class is imported
from option.train_option import get_train_options  # Import separate option configuration
import logging
import os
from Util.tools import adjust_lr
from torch.utils.data import DataLoader

# Get training parameters
opt = get_train_options()

# Set GPU (if needed)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

# Model save path configuration (using settings from opt)
save_model_path = opt.save_model_path
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

# Logging configuration
log_path = opt.log_path
if not os.path.exists(log_path):
    os.makedirs(log_path)

file_handler = logging.FileHandler(os.path.join(log_path, 'log_model_train.log'), mode='a', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', datefmt='%Y-%m-%d %I:%M:%S %p')
file_handler.setFormatter(formatter)
logging.getLogger('').addHandler(file_handler)
logging.getLogger('').setLevel(logging.INFO)
logging.info("Model-Train")
logging.info("Configuration Parameters")
# Print all configuration parameters
for arg in vars(opt):
    logging.info(f'{arg}: {getattr(opt, arg)}')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
logging.info(f'Using device: {device}')

# Model initialization
generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator_ct_feature = Generator_Ct_Feature().to(device)  # Initialize CT feature extractor separately

# Loss function definition
criterion_GAN = torch.nn.BCEWithLogitsLoss()  # Adjust GAN loss as needed
criterion_L1 = torch.nn.L1Loss()
criterion_MSE = torch.nn.MSELoss()
vggLoss = VGGLoss().to(device) if cuda else VGGLoss()

# Move models and loss functions to GPU
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    generator_ct_feature = generator_ct_feature.cuda()
    criterion_GAN.cuda()
    criterion_L1.cuda()
    criterion_MSE.cuda()

# Loss recording lists
generator_losses = []
discriminator_losses = []

# Optimizer configuration (using learning rate from opt)
optimizer_G = torch.optim.Adam(
    list(generator.parameters()) + list(generator_ct_feature.parameters()),
    lr=opt.lr,
    betas=(0.5, 0.999),
    weight_decay=0.0001
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=opt.lr * 0.05,  # Discriminator learning rate is 1/20 of Generator's
    betas=(0.5, 0.999),
    weight_decay=0.0001
)
logging.info(f'Generator Optimizer: lr={opt.lr}, betas=(0.5, 0.999), weight_decay=0.0001')
logging.info(f'Discriminator Optimizer: lr={opt.lr * 0.05}, betas=(0.5, 0.999), weight_decay=0.0001')

# Data path configuration (using settings from opt)
source_image_root = opt.source_image_root
target_image_root = opt.target_image_root
mask_root = opt.mask_root

# Data transformations
transforms_ = [
    transforms.Resize((opt.trainsize, opt.trainsize), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Data loader (using batch_size and num_workers from opt)
dataloader = DataLoader(
    ImageDatasetV2(opt.dataset_name, transforms_=transforms_, mode='train'),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    pin_memory=True if cuda else False
)
total_step = len(dataloader)
# Adapt discriminator output size (calculated based on trainsize)
patch_size = (1, opt.trainsize // 2 ** 4, opt.trainsize // 2 ** 4)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Best model recording parameters
best_g_loss = float('inf')
best_epoch = 0


def normalize_output(data):
    """Data normalization (mimicking limit_zero2one functionality, adjust as needed)"""
    data = data.clone()
    data = torch.clamp(data, -1.0, 1.0)
    return (data + 1.0) / 2.0


def train(dataloader, generator, discriminator, generator_ct_feature, epoch, save_path, optimizer_g, optimizer_d):
    """Training function, maintaining the same process structure as the original file, using loss weights from opt"""
    global best_epoch, best_g_loss
    generator.train()
    discriminator.train()
    generator_ct_feature.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    for i, batch in enumerate(dataloader):
        # Zero gradients
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        # Move data to GPU
        source_imgs = Variable(batch["A"].type(Tensor)).to(device)
        target_imgs = Variable(batch["B"].type(Tensor)).to(device)
        masks = Variable(batch["C"].type(Tensor)).to(device)

        # Adversarial labels
        valid = Variable(Tensor(np.ones((target_imgs.size(0), *patch_size))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((source_imgs.size(0), *patch_size))), requires_grad=False)

        # ---------------------
        #  Train Generator
        # ---------------------
        guided_CT, pet_commonFeature, gen_CT, seg = generator(source_imgs)  # Generator with mask input
        ct_commonFeature = generator_ct_feature(target_imgs)

        # Calculate each loss (using weights from opt)
        loss_feature = criterion_L1(pet_commonFeature, ct_commonFeature) * opt.feature_weight
        loss_mid = criterion_MSE(guided_CT, target_imgs) * opt.mid_weight

        # GAN loss
        pred_fake = discriminator(source_imgs, gen_CT)
        loss_gan = criterion_GAN(pred_fake, valid) * opt.gan_weight

        # Pixel loss
        loss_pix = criterion_L1(gen_CT, target_imgs) * opt.pix_weight

        # Structure loss
        loss_str = criterion_MSE(seg, masks) * opt.str_weight

        # Texture loss
        loss_tex = vggLoss(gen_CT, target_imgs) * opt.tex_weight

        # Total Generator loss
        g_loss = loss_gan + loss_pix + loss_str + loss_feature + loss_mid + loss_tex
        g_loss.backward()
        optimizer_g.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Real sample loss
        pred_real = discriminator(source_imgs, target_imgs)
        loss_d_real = criterion_GAN(pred_real, valid)

        # Fake sample loss (fixed gen_output undefined error from original code)
        pred_fake = discriminator(source_imgs, gen_CT.detach())
        loss_d_fake = criterion_GAN(pred_fake, fake)

        # Total Discriminator loss
        d_loss = (loss_d_real + loss_d_fake) * 0.5
        d_loss.backward()
        optimizer_d.step()

        # Accumulate losses
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

        # Print training progress (fixed loss_content and loss_perceptual undefined error from original code)
        print(
            '#TRAIN#{}:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], GAN_loss: {:.4f}, pix_loss: {:.4f}, tex_loss: {:.4f}, G_all_loss: {:.4f}, D_loss: {:.4f}'.
            format(datetime.now(), epoch, opt.n_epochs, i, total_step, loss_gan.item(), loss_pix.item(),
                   loss_tex.item(), g_loss.item(), d_loss.item())
        )
        logging.info(
            '#TRAIN#{}:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], GAN_loss: {:.4f}, pix_loss: {:.4f}, tex_loss: {:.4f}, G_all_loss: {:.4f}, D_loss: {:.4f}'.
            format(datetime.now(), epoch, opt.n_epochs, i, total_step, loss_gan.item(), loss_pix.item(),
                   loss_tex.item(), g_loss.item(), d_loss.item())
        )

    # Record epoch total loss
    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)

    logging.info("\n=============================================")
    logging.info(f'Epoch {epoch} Generator average loss: {avg_g_loss:.10f}')
    logging.info(f'Epoch {epoch} Discriminator average loss: {avg_d_loss:.10f}')
    logging.info("=============================================\n")

    # Save best model
    if avg_g_loss < best_g_loss:
        best_g_loss = avg_g_loss
        best_epoch = epoch
        torch.save(generator.state_dict(), os.path.join(save_path, f"Best_Generator_{epoch}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_path, f"Best_Discriminator_{epoch}.pth"))
        torch.save(generator_ct_feature.state_dict(), os.path.join(save_path, f"Best_Generator_CT_Feature_{epoch}.pth"))
        logging.info(f'#SAVE# Best model updated at Epoch {epoch}, Generator loss: {best_g_loss:.6f}')

    # Save model periodically by frequency
    if epoch % opt.save_freq == 0:
        torch.save(generator.state_dict(), os.path.join(save_path, f"Generator_Epoch_{epoch}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(save_path, f"Discriminator_Epoch_{epoch}.pth"))
        torch.save(generator_ct_feature.state_dict(),
                   os.path.join(save_path, f"Generator_CT_Feature_Epoch_{epoch}.pth"))
        logging.info(f'#SAVE# Periodically save model at Epoch {epoch}')


if __name__ == '__main__':
    print("Starting model training...")
    # Fix random seed (using seed from opt)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logging.info(f'Random seed set to: {opt.seed}')
    logging.info(f'Total training epochs: {opt.n_epochs}')
    logging.info(f'Batch size: {opt.batch_size}')
    logging.info(f'Dataset size: {len(dataloader.dataset)}')
    logging.info(f'Total steps per epoch: {total_step}')

    # Training main loop
    for epoch in range(1, opt.n_epochs + 1):
        current_lr = adjust_lr(optimizer_G, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        logging.info(f'Epoch {epoch} Learning rate: {current_lr:.6f}')
        train(
            dataloader,
            generator,
            discriminator,
            generator_ct_feature,  # Pass CT feature extractor
            epoch,
            save_model_path,
            optimizer_G,
            optimizer_D
        )

    # Print best model information after training completion
    logging.info("\n=============================================")
    logging.info(f'Training completed! Best model located at Epoch {best_epoch}, Best Generator loss: {best_g_loss:.6f}')
    logging.info("=============================================\n")
    print(f"Training completed! Best model located at Epoch {best_epoch}, Best Generator loss: {best_g_loss:.6f}")