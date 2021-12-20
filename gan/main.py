import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import gan.gan as gan
import gan.gan_sample_generation as gener
import gan.sample_load_gan_results_yy as csv_gener
import common.r2_summary_gan as r2
import shutil
import os

types = [9, 5]
gens = [200, 250]
if not os.path.isdir('./h5'):
    os.mkdir('./h5')
if not os.path.isdir('./results'):
    os.mkdir('./results')

for i in range(1,11):
    num_train = [i*10, i*10]
    gan.main(types, gens, num_train)
    print('complete_'+str(i)+'_gan_training')
    
    for j in range(1,2):
        if not os.path.isdir('./'+str(j)+'_csv'):
            os.mkdir('./'+str(j)+'_csv')
        gener.main(types, gens, num_train)
        print('complete_'+str(i)+'_gan_generation_pickle')
        
        csv_gener.main(types, [num_train[0]])
        print('complete_'+str(i)+'_gan_generation_csv')
        
        r2.main(types, [num_train[0]], j)
        print('complete_'+str(i)+'_gan_r2')
        shutil.rmtree('./results')
        os.mkdir('./results')
        print('done')
