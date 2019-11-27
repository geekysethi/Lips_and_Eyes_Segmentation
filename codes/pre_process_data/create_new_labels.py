import numpy as np
import cv2

labels_folder = "C:\\Users\\ashishHP\Desktop\\fynd_assignment\\raw_data\\CelebAMask-HQ\\CelebAMask-HQ-mask-anno"
save_path = "C:\\Users\\ashishHP\\Desktop\\fynd_assignment\\raw_data\\new_labels"
total_images = 30000

components = ["l_eye","r_eye","l_lip","u_lip"]
for i in range(total_images):
	# print("***************")
	print(i)
	final_image = np.zeros((512,512,3))
	current_folder = labels_folder + "\\" + str(i//2000)
	
	for current_component in components:
		current_image_id = "{}".format(str(i).zfill(5))
	
		current_image_id = current_image_id +"_"+current_component + ".png"
		current_image_path = current_folder + "\\" + current_image_id
		# print(current_image_path)
		try:
			current_image = cv2.imread(current_image_path)
			# print(current_image.shape)
			final_image +=current_image
			final_image = final_image.astype("uint8")
			final_image[final_image<127] = 0
			final_image[final_image>=127] = 255
			
		except:
			print("EXCEPTION")
			
		
	save_image_path = save_path+ "\\"+str(i)+".png"
	cv2.imwrite(save_image_path,final_image)