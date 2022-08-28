# Face Smile Detection

[![Face Smile Detection](https://img.youtube.com/vi/eC_GfTEylSw/0.jpg)](https://www.youtube.com/watch?v=eC_GfTEylSw)

## follow these step to run this application 
 
1. Clone code  and open code in VS code (recommended)
2. create a virtual environment 
   **macos/linux** : `python3 -m venv venv`
   **winodws** : `py -3 -m venv venv`
3. Activate virtual environment 
   **macos/linux** :  `. venv/bin/activate`
   **windows** : `venv\Scripts\activate`
   
4. install all python package 
   `pip install -r requirements.txt`
5. command for training your model
   `python train.py --dataset ./datasets/smileD  --model ./output/lenet.hdf5`
6. command for detect face smile 
    `python detect_smile.py --cascade haarcascade_frontalface_default.xml  --model ./output/lenet.hdf5`
