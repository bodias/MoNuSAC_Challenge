from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import cv2
import os


class Challenge(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, img_dir, mask_dir, input_img_paths, target_img_paths, shuffle=False, augment=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.shuffle = shuffle
        self.augment = augment
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        
        for j, (img_path, mask_path) in enumerate(zip(batch_input_img_paths, batch_target_img_paths)):
            img = load_img(os.path.join(self.img_dir, img_path), target_size=self.img_size)
            #mask = load_img(mask_path, target_size=self.img_size, color_mode="grayscale")
            #img = cv2.imread(os.path.join(self.img_dir, img_path))
            #img = img[:self.img_size[0], :self.img_size[1], :]
            mask = np.load(os.path.join(self.mask_dir, mask_path))
            if (mask.shape[0],mask.shape[1]) != self.img_size:
                mask = cv2.resize(mask, dsize=self.img_size, interpolation=cv2.INTER_NEAREST_EXACT)
            #mask = mask[:self.img_size[0], :self.img_size[1]]
            if self.augment:
                img, mask = self.augmentation(img, mask, p=.6)
            #mask = self._preprocess_mask(mask)
            mask = np.expand_dims(mask, 2)
            
            x[j] = img            
            y[j] = mask
        
        return x, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.input_img_paths)
            np.random.shuffle(self.target_img_paths)
    
    def _preprocess_mask(self, mask_img):
        mask = np.array(mask_img)
        mask[mask>0] = 1
        return mask
    
    def augmentation(self, img, mask, p=0.5):
        im_size = img.size
        
        # random rotate
        if np.random.rand()<p:
            angle = np.random.randint(1,360)
            img = img.rotate(angle, expand=True)
            mask = mask.rotate(angle, expand=True)

        # random crop
        if np.random.rand()<p:
            # generate 4 random numbers to crop the image in all 4 sides
            w, h = img.size
            max_cut_w = int(w * 0.2) #cut max 20%
            max_cut_h = int(h * 0.2) #cut max 20%
            cut_w = np.random.randint(1, max_cut_w, size=2)
            cut_h = np.random.randint(1, max_cut_h, size=2)

            img = img.crop((cut_w[0], cut_h[0], w - cut_h[1], w - cut_h[1]))
            mask = mask.crop((cut_w[0], cut_h[0], w - cut_h[1], w - cut_h[1]))

        assert img.size==mask.size, "something wrong with crop"
        
        # put back to original size in case something has changed
        if img.size!= im_size:
            img = img.resize(im_size)
            mask = mask.resize(im_size)

        return img, mask