import gan_final_generations as gener
import gan_to_csv as csv_gener
import os

types = [7, 12, 14, 15]
gens = [400, 4500, 800, 400]

if not os.path.isdir('./results'):
    os.mkdir('./results')

for i in range(10,11):
    num_train = [i*10, i*10, i*10, i*10]
    # gan.main(types, gens, num_train)
    
    gener.main(types, gens, num_train)
        
    csv_gener.main(types, [num_train[0]])
    print('done')
