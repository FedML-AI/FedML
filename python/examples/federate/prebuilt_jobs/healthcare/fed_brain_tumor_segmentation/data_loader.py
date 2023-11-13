from data_downloader import download_data

import numpy as np
import logging
import os
import cv2
import glob
import shutil

def load_mask(mask_dir, mask_list):
    masks = []
    mask_list.sort()

    for mask_name in mask_list:
        path = os.path.join(mask_dir,mask_name)
        mask = np.load(path)
        masks.append(mask)

    masks = np.array(masks)
    masks = np.stack(masks, axis=0)
    masks = np.transpose(masks, (0, 2, 3, 1))
    return(masks)

def normalizing_images(trainImages):
  normalized_images = []

  for image in trainImages:

      min_val = image.min()
      max_val = image.max()

      normalized_image = (image - min_val) / (max_val - min_val) + 1e-7

      normalized_images.append(normalized_image)

  trainImages = np.array(normalized_images)
  return trainImages

from sklearn.preprocessing import LabelEncoder
import numpy as np

def reEncodeChannel(masks):
    # Initialize LabelEncoder
    labelencoder = LabelEncoder()

    # Get the dimensions of masks
    masksNum, channels, height, width = masks.shape

    # Initialize an array to store the encoded masks
    masks_encoded = np.empty_like(masks, dtype=int)

    # Loop through each mask and channel
    for mask_idx in range(masksNum):
        for channel_idx in range(channels):
            # Get the current channel
            current_channel = masks[mask_idx, channel_idx]

            # Reshape the current channel to (256*256, 1)
            current_channel_reshaped = current_channel.reshape(-1, 1)

            # Encode the current channel
            current_channel_encoded = labelencoder.fit_transform(current_channel_reshaped.ravel())

            # Reshape the encoded channel back to its original shape
            current_channel_encoded_original_shape = current_channel_encoded.reshape(height, width)

            # Check for values higher than 1 and set them to 1
            current_channel_encoded_original_shape[current_channel_encoded_original_shape > 1] = 1

            # Store the encoded channel in the corresponding position
            masks_encoded[mask_idx, channel_idx] = current_channel_encoded_original_shape

    return masks_encoded

def load_img(img_dir, img_list):
    images = []
    img_list.sort()

    for image_name in img_list:
        path = os.path.join(img_dir,image_name)
        image = cv2.imread(path,1) # 1 for colored images
        images.append(image)

    images = np.array(images)

    return(images)

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    img_list.sort()
    mask_list.sort()

    dataset_length = len(img_list)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < dataset_length: # The dataset has not ended
            limit = min(batch_end, dataset_length) # Get the last batch

            images = load_img(img_dir, img_list[batch_start:limit]).astype(np.float32) # load x
            masks = load_mask(mask_dir, mask_list[batch_start:limit]).astype(np.float32) # load y

            images = normalizing_images(images)
            masks = reEncodeChannel(masks)

            yield (images,masks)

            # Get the next batch
            batch_start += batch_size
            batch_end += batch_size
            # FIXME (Mariem Abdou) remove break - it seems that your program hangs!, that's why it was added
            break
        # FIXME (Mariem Abdou) remove break - it seems that your program hangs!, that's why it was added
        break

# COPY AS IS, EDIT THE PATHS THE FOLDERS NAME IN download_datas
def load_data(args):
    train_data_num = 0
    test_data_num = 0
    train_data_global = None
    test_data_global = None
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    nc = args.output_dim

    if args.process_id == 0:  # server
        logging.info("load data for server test")
        download_data(file_name = "Test10.zip",
                      folder_name = "TEST",
                      output_folder = "Test",
                      remove_zip = True
                      )

        TEST_DATA_DIR = "/content/Test"

        BATCH_SIZE_TEST = args.batch_size

        test_img_list = sorted(glob.glob(os.path.join(TEST_DATA_DIR, '**/*.png'), recursive=True))
        test_mask_list = sorted(glob.glob(os.path.join(TEST_DATA_DIR, '**/*.npy'), recursive=True))

        test_generator = imageLoader(TEST_DATA_DIR, test_img_list,
                                TEST_DATA_DIR, test_mask_list, BATCH_SIZE_TEST)

        logging.info("server - test_generator created")

        client_idx = int(args.process_id) - 1
        test_data_num = len(test_img_list)
        test_data_global = test_generator
        test_data_local_dict[client_idx] = test_generator
        logging.info("server - Afterrrrr test_generator created")
        return (
            train_data_num, # 0 for server
            test_data_num, # lenght of the test data for server
            train_data_global, # None for server
            test_data_global, # imageLoader(test)
            train_data_local_num_dict, # None for server
            train_data_local_dict, # None for server
            test_data_local_dict, # None for server
            nc, # args.output_dim
        )
    else:  # client
        client_idx = int(args.process_id) - 1
        logging.info(f"load center {int(client_idx)} data")
        # DOWNLOAD DATASET FOR EACH CLIENT
        if client_idx==0:
            download_data(file_name = "client 1.zip",
              folder_name = "Clients",
              output_folder = "client1",
              remove_zip = True
              )
            args.data_cache_dir = os.path.join(os.getcwd(), 'client1')#EDIT

        elif client_idx==1:
            download_data(file_name = "client 2.zip",
              folder_name = "Clients",
              output_folder = "client2",
              remove_zip = True
              )
            args.data_cache_dir = os.path.join(os.getcwd(), 'client2')#EDIT

        elif client_idx==2:
            download_data(file_name = "client 3.zip",
              folder_name = "Clients",
              output_folder = "client3",
              remove_zip = True
              )
            args.data_cache_dir = os.path.join(os.getcwd(), 'client3')#EDIT

        elif client_idx==3:
            download_data(file_name = "client 4.zip",
              folder_name = "Clients",
              output_folder = "client4",
              remove_zip = True
              )
            args.data_cache_dir = os.path.join(os.getcwd(), 'client4')#EDIT

        TRAIN_DATA_DIR = args.data_cache_dir
        # data_dir_train = os.path.join(data_dir, 'train')#EDIT
        # data_dir_train_image = os.path.join(data_dir_train, 'imagesTr', "")#EDIT
        # data_dir_train_mask = os.path.join(data_dir_train, 'labelsTr', "")#EDIT

        # data_dir_val = os.path.join(data_dir, 'val')#EDIT
        # data_dir_val_image = os.path.join(data_dir_val, 'imagesTr', "")#EDIT
        # data_dir_val_mask = os.path.join(data_dir_val, 'labelsTr', "")#EDIT

        BATCH_SIZE_TRAIN = args.batch_size
        BATCH_SIZE_TEST = args.batch_size

        train_img_list= sorted(glob.glob(os.path.join(TRAIN_DATA_DIR, '**/*.png'), recursive=True))
        train_mask_list = sorted(glob.glob(os.path.join(TRAIN_DATA_DIR, '**/*.npy'), recursive=True))

        train_generator = imageLoader(TRAIN_DATA_DIR, train_img_list,
                                        TRAIN_DATA_DIR, train_mask_list, BATCH_SIZE_TRAIN)

        logging.info("client - train_generator created")

        # val_img_list= sorted(os.listdir(data_dir_val_image))
        # val_mask_list = sorted(os.listdir(data_dir_val_mask))

        # val_generator = imageLoader(data_dir_val_image, val_img_list,
                                    # data_dir_val_mask, val_mask_list, BATCH_SIZE_TRAIN)

        # test_img_list= sorted(os.listdir(data_dir_test_image))
        # test_mask_list = sorted(os.listdir(data_dir_test_mask))

        # test_generator = imageLoader(data_dir_test_image, test_img_list,
        #                             data_dir_test_mask, test_mask_list, BATCH_SIZE_TEST)

        # logging.info("client - test_generator created")

        train_data_num = len(train_img_list)
        # FIXME (Mariem Abdou) replace with actual test
        test_data_num = len(train_img_list)
        train_data_global = train_generator
        # FIXME (Mariem Abdou) replace with actual test
        test_data_global = train_generator
        train_data_local_num_dict[client_idx] = train_data_num
        train_data_local_dict[client_idx] = train_generator
        # FIXME (Mariem Abdou) replace with actual test
        test_data_local_dict[client_idx] = train_generator
        logging.info(f"train_data_num: {train_data_num}")
        logging.info(f"test_data_num: {test_data_num}")
        logging.info(f"train_data_global: {train_data_global}")
        logging.info(f"test_data_global: {test_data_global}")
        logging.info(f"data_local_num_dict: {train_data_local_num_dict}")
        logging.info(f"train_data_local_dict: {train_data_local_dict}")
        logging.info(f"test_data_local_dict: {test_data_local_dict}")
        logging.info(f"nc: {nc}")

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        nc,
    )