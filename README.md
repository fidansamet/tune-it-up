To train the best model:

    python train.py --dataroot ./datasets/J2C --name j2c --activation sigmoid --preprocess tensor --model cycle_gan --no_flip --triplet --mixed_disc
    
To test the best model:

    python test.py -dataroot ./datasets/J2C --name j2c --activation sigmoid --preprocess tensor --model cycle_gan --no_flip --triplet --mixed_disc --phase test --no_dropout

To evaluate the best model:

    python genre_classifier.py checkpoints/j2c/npy/ Classic
