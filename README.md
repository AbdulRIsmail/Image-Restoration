# 🌃 🔨 Image Restoration
Restores 2 types of images (border & damage) with image processing using OpenCV

Before             |  After
:-------------------------:|:-------------------------:
![Faded](https://user-images.githubusercontent.com/39646629/154096273-1d7001de-8a09-4b4a-b127-a2c79b2d4d04.jpg) | ![Faded-Fix](https://user-images.githubusercontent.com/39646629/154096288-57019e12-f416-4bb1-b903-f07021acb144.png)

Before             |  After
:-------------------------:|:-------------------------:
<img height="344" src="https://user-images.githubusercontent.com/39646629/154096754-e942f016-f231-4928-9b98-ecf486b2caf9.jpg" /> | <img height="344" src="https://user-images.githubusercontent.com/39646629/154096761-e07c828b-3687-4985-ac5b-672c13d34032.png" />

<br>

## 🤨 How It Works
### Description  
In this program I have 2 algorithms (faded & damaged) to restore images. The program knows what algorithm to run using the find_image_algorithm. I will explain this algorithm in the section below. 

### Faded Image - Algorithm   
The faded algorithm works by firstly finding the thresh of the rectangle border. I used canny to get the edges and then added a threshold to make it cleaner. I found the contours of the canny + threshold and drew it on a empty black canvas. I created horizontal and vertical kernels to find the rectangle borders. Once I got the border, I got the 4 points (top, left, bottom, right). I then cropped out the faded outline using the points and added a darker contrast and then added it back to the image. This leaves a very harsh rectangle line. I finally used inpaint to get the neighbouring regions and blend the harsh line.

### Damaged Image - Algorithm  
The damaged algorithm works by using the denoise function provided by cv2. The arguments passed into the denoise function removes the noise while also trying to keep the sharpness of the image.

### find_image_algorithm  
The find_image_algorithm works by calculating the image histogram. This will return 3 values BGR (Blue, Green, Red). I ravel each of the values and then zip them. I then sort the zipped values in reverse. I then find the x_axis index for the BGR values and store it. I then finally return the maximum values of B, G, R. If the maximum value is greater than 110, this means the image is colored so run the faded algorithm otherwise it will run the damaged algorithm.
<br>
## Other Image You Can Test Out!
<img height="344" src="https://user-images.githubusercontent.com/39646629/154096345-9092a4d7-f574-4b03-9f00-0becb862830b.jpg" />
