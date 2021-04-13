import gan
import cgan
import cgan_sample_generation as gener
import sample_load_cgan_results as csv_gener
import r2_summary_gan as r2
import shutil
import os

types = [7, 12, 14, 15]
gens = [400, 4500, 800, 400]
if not os.path.isdir('./h5'):
    os.mkdir('./h5')
if not os.path.isdir('./results'):
    os.mkdir('./results')

for i in range(1, 11):
    num_train = i*10
    # cgan.main(types, num_train)
    print('complete_'+str(i)+'_gan_training')

    for j in range(1, 6):
        if not os.path.isdir('./'+str(j)+'_csv'):
            os.mkdir('./'+str(j)+'_csv')
        # gener.main(types, gens, num_train)
        print('complete_'+str(i)+'_gan_generation_pickle')

        csv_gener.main(types, num_train)
        print('complete_'+str(i)+'_gan_generation_csv')

        r2.main(types, [num_train[0]], j)
        print('complete_'+str(i)+'_gan_r2')
        shutil.rmtree('./results')
        os.mkdir('./results')
        print('done')
