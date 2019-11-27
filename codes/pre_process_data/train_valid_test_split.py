import numpy as np 
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

def save_images(image_list, split):

    save_images_path = save_path + "\\" + str(split) +"\\images"
    save_labels_path = save_path + "\\" + str(split) +"\\labels"
    
    columns = ["image_id"]
    for current_image in image_list:

        print(current_image)

        shutil.copy(data_images_path+"\\"+str(current_image)+".jpg",save_images_path+"\\"+str(current_image)+".jpg",)
        shutil.copy(data_labels_path+"\\"+str(current_image)+".png",save_labels_path+"\\"+str(current_image)+".png",)


    print("[INFO] SAVING DATA IN DATAFRAME")
    df = pd.DataFrame(index=np.arange(len(image_list)), columns=columns)
    df.loc[:] = np.reshape(image_list,(len(image_list),1))
    df.to_csv(save_path + "\\" + str(split) +"\\data.csv", encoding="utf-8", index=False)
    
save_path = "C:\\Users\\ashishHP\\Desktop\\fynd_assignment\\data"
data_images_path = "C:\\Users\\ashishHP\\Desktop\\fynd_assignment\\raw_data\\CelebAMask-HQ\\CelebA-HQ-img"
data_labels_path = "C:\\Users\\ashishHP\Desktop\\fynd_assignment\\raw_data\\new_labels"

random_seed = 42
test_split = .30
validation_split = .20
total_images = 30000
total_images_list = np.arange(total_images)

train_images_list,validation_images_split = train_test_split(total_images_list,test_size = validation_split,random_state = random_seed)

train_images_list,test_images_split = train_test_split(train_images_list,test_size = test_split,random_state = random_seed)

print(len(train_images_list))
print(len(validation_images_split))
print(len(test_images_split))

# save_images(train_images_list,"train")
save_images(validation_images_split,"validation")
save_images(test_images_split,"test")
