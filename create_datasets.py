#done in project folder containing PlantVillage
import shutil, os, random

#function to move images
def move_images(image_set, dataset):
    for image in image_set:
        original_image_path = class_path + '/' + image
        new_image_path = dataset + '/' + category + '/' + image
        os.rename(original_image_path,new_image_path)

classes = os.listdir('PlantVillage')
classes.remove('.DS_Store')


for category in classes:
    total = len(os.listdir('PlantVillage' + '/' + category))
    train_amount = int(0.68*total)
    valid_amount = int(0.22*total)
    test_amount = total - train_amount - valid_amount
    class_path = 'PlantVillage/' + category

    #import pdb; pdb.set_trace()
    train_images = random.sample(os.listdir(class_path),train_amount)
    move_images(train_images,'Training_Set')
    valid_images = random.sample(os.listdir(class_path),valid_amount)
    move_images(valid_images,'Validation_Set')
    test_images = random.sample(os.listdir(class_path),test_amount)
    move_images(test_images,'Testing_Set')
