from data import DIV2K

from model.srgan import generator, discriminator
from train import SrganGeneratorTrainer, SrganTrainer


train_loader = DIV2K(scale=4,             # 2, 3, 4 or 8
                     downgrade='bicubic',  # 'bicubic', 'unknown', 'mild' or 'difficult'
                     subset='train')      # Training dataset are images 001 - 800

# Create a tf.data.Dataset
train_ds = train_loader.dataset(batch_size=16,         # batch size as described in the EDSR and WDSR papers
                                random_transform=True,  # random crop, flip, rotate as described in the EDSR paper
                                repeat_count=None)     # repeat iterating over training images indefinitely

valid_loader = DIV2K(scale=4,             # 2, 3, 4 or 8
                     downgrade='bicubic',  # 'bicubic', 'unknown', 'mild' or 'difficult'
                     subset='valid')      # Validation dataset are images 801 - 900

# Create a tf.data.Dataset
valid_ds = valid_loader.dataset(batch_size=1,           # use batch size of 1 as DIV2K images have different size
                                random_transform=False,  # use DIV2K images in original size
                                repeat_count=1)         # 1 epoch

# Create a training context for the generator (SRResNet) alone.
pre_trainer = SrganGeneratorTrainer(model=generator(),
                                    checkpoint_dir='.ckpt/pre_generator')

# Pre-train the generator with 1,000,000 steps (100,000 works fine too).
pre_trainer.train(train_ds, valid_ds.take(10),
                  steps=1000000, evaluate_every=1000)

# Save weights of pre-trained generator (needed for fine-tuning with GAN).
pre_trainer.model.save_weights('weights/srgan/pre_generator.h5')

gan_trainer = SrganTrainer(generator=pre_trainer.model,
                           discriminator=discriminator())

gan_trainer.train(train_ds, steps=200000)

gan_trainer.generator.save_weights('weights/srgan/gan_generator.h5')
gan_trainer.discriminator.save_weights('weights/srgan/gan_discriminator.h5')
