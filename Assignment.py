import cv2
import numpy as np
import easygui
import matplotlib.pyplot as plt

f = easygui.fileopenbox()
img = cv2.imread(f)

def faded(image):
    """
    Method: algorithm to fix the faded image
    Returns: the fixed image
    """

    image_copy = image.copy() # copy image
    h, w, d = image.shape # get height, width, depth of the image
    copy_h, copy_w = h, w # make a copy of the height and width

    def find_thresh():
        """
        Method: Finds the thresh of the image. It creates an empty black canvas and uses canny to find the edges.
                Once the edges are found, it runs through threshhold to make it clearer.
                It then finds the contours of the image using the thresh.

        Returns: the drawn contours of the image on the black canvas.
        """

        # creates an empty black canvas (same size as the original image)
        # this will be used to draw the contours on
        black_canvas = np.zeros((h, w, d), np.uint8)

        # uses canny to detect edges on the image - this will find the edges
        edge = cv2.Canny(image, 90, 170)

        # uses threshold with the canny edges to make it clearer - it uses  a light value (127)
        thresh = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)[1]

        # finds all the contours of the threshold
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

        # draws white contours on the empty blank canvas
        # setting the stroke to 3 will lead to easier border detection
        return cv2.drawContours(black_canvas, contours, -1, (255, 255, 255), 3)

    def create_kernels():
        """
        Method: creates the horizontal and vertical kernels
        Returns: the horizontal & vertical kernel
        """

        # creates the horizontal kernel - this will be used to find the horizontal lines in the faded border
        horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))

        # creates the vertical kernel - this will be used to find the vertical lines in the faded border
        vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))

        # Returns the horizontal & vertical kernel
        return horizontal, vertical

    def kernel_lines(horizontal, vertical, canvas):
        """
        Method: finds the horizontal & vertical lines on the canvas (the lines of the faded border)
        Params:
            horizontal: the horizontal kernal
            vertical: the vertical kernal
            canvas: the canvas that has the threshold and contours drawn from findThresh()
        Returns: the horizontal & vertical lines
        """

        # finds the horizontal lines in the canvas border using morphologyEx
        horizontal_lines = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, horizontal)

        # finds the vertical lines in the canvas border using morphologyEx
        vertical_lines = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, vertical)

        # the horizontal & vertical lines
        return horizontal_lines, vertical_lines

    def find_borders(canvas):
        """
        Method: finds the full rectangle border of the damaged faded image
        Params:
            canvas: the canvas that has the threshold and contours drawn from findThresh()
        Returns:
            Returns the faded border outline contours found in the gray image
            using (RETR_EXTERNAL) to retrieve the extreme outer contours
            using (CHAIN_APPROX_SIMPLE) to compress horizontal & vertical to only leave their end points
            [0] at the end to only get the contour and not the hierarchy
         """

        # gets the horizontal & vertical kernal from create_kernels()
        horizontal, vertical = create_kernels()

        # gets the horizontal & vertical lines from kernel_lines()
        horizontal_lines, vertical_lines = kernel_lines(horizontal, vertical, canvas)

        # add the horizontal & vertical lines using bitwise_and to form a rectangle border
        # bitwise_and is used to find both the horizontal & vertical lines which are ON
        border = cv2.bitwise_and(horizontal_lines, vertical_lines)

        # change the border to gray to use findContours
        border_gray = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)

        # returns the faded rectangle border contours found in the gray image
        return cv2.findContours(border_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    def find_border_points(contours):
        """
        Method: finds the 4 rectangle (top, left, bottom, right) border points of the contours
        Params:
            contours: has the faded rectangle border outline of the image from find_borders()
        Returns: the 4 points (top, left, bottom, right) of the faded border outline
        """

        border_points = [] # stores the border points
        for c in contours: # loop through contours
            (x, y, w, h) = cv2.boundingRect(c) # get the bounding rect for each counter
            border_points.append([x,y, x+w,y+h]) # appends the intersecting points

        left, top = np.min(border_points, axis=0)[:2] # gets the left and top points
        right, bottom = np.max(border_points, axis=0)[2:] # gets the right and bottom points

        # returns the 4 points of the border outline
        return top, left, bottom, right

    def fix_image(top_left, bottom_right):
        """
        Method: Fixes the faded image. It does so by cropping out the faded border section
                using the 4 border points from the original image. I then add a black contrast to the cropped canvas
        Params:
            top_left: the top left point of the border
            bottom_right: the bottom right point of the border
        """

        # gets a cropped section of the original image using the top_left, bottom_right points
        cropped_image = image_copy[4 + top_left[0]: - 3 + bottom_right[0], 2 + top_left[1]: -2 + bottom_right[1]]

        # store height, width, depth of the cropped image
        crop_h, crop_w, crop_d = cropped_image.shape

        # dark contrast to add to the black canvas
        dark_contrast = 33

        # create black canvas using the cropped image size
        # this canvas now stores the faded border and we add the dark_contrast to make it darker
        fixed_cropped_image_canvas = np.zeros((crop_h, crop_w, crop_d), np.uint8) + dark_contrast

        # subtracting the cropped_image from the fixed_cropped_image_canvas
        fixed_border_image = cv2.subtract(cropped_image, fixed_cropped_image_canvas)

        # setting the faded border region in the image_copy to the fixed border region
        image_copy[4 + top_left[0]: - 3 + bottom_right[0], 2 + top_left[1]: -2 + bottom_right[1]] = fixed_border_image

    def blend_image(top, left, bottom, right):
        """
        Method: blends the harsh rectangle border using inpaint
        Params:
            top: the top point of the border
            left: the left point of the border
            bottom: the bottom point of the border
            right: the right point of the border
        """

        # fixes the image (changes the image_copy)
        fix_image((top, left), (bottom, right))

        # creating black canvas the same size of the original image
        # this is used to blend the harsh border lines when the image is fixed
        blend_canvas = np.zeros((copy_h, copy_w, d), np.uint8)

        # draw a rectangle on the blend canvas using the 4 points (top, left, bottom, right)
        # fill the rectangle to white with a stroke of 7 - this helps to get a better region neighbourhood for inpaint
        cv2.rectangle(blend_canvas, (left, top), (right, bottom), (255, 255, 255), 7)

        # change the blend canvas to gray
        blend_gray = cv2.cvtColor(blend_canvas, cv2.COLOR_BGR2GRAY)

        # use inpaint to restore the selected region in the image using the region neighbourhood
        return cv2.inpaint(image_copy, blend_gray, 3, cv2.INPAINT_TELEA)

    canvas = find_thresh() # finds the thresh
    contours = find_borders(canvas) #  finds the border contours
    top, left, bottom, right = find_border_points(contours) # get the faded rectangle border points
    return blend_image(top, left, bottom, right) # return the fixed image


def damaged(image):
    """
    Method: fixes the damaged image by using fastNlMeansDenoisingColored
            this will remove all the noise from the image and smoothen it
    Returns: the fixed image
    """

    image_copy = image.copy() # copy image
    denoised_image = cv2.fastNlMeansDenoisingColored(image_copy, None, 9, 9, 7, 21)

    # return the denoised_image in gray
    return cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)

def find_image_algorithm(img):
    """
    Method: finds which fixing algorithm to use depending on the image histogram by finding
            the highest spike (BLUE, GREEN, RED) in the x_axis
    Returns: the fixed image
    """

    def maximum(blue, green, red):
        """
        Method: finds the maximum x_axis of the blue, green, red histogram
        Returns: the maximum x_axis
        """
        return max([blue, green, red])

    def find_highest_spike():
        indices = list(range(0, 256)) # make a list of range 0 to 256

        # set all variables to Array of size 3 and fill it with None
        # the size is set to 3 because of BGR (Blue, Green, Red)

        # colors - used to store the calcHistogram of each BGR
        # colors_val - used to store the ravel of each BGR histogram
        # colors_zipped - used to store the zipped value of colors_val
        # colors_index - used to store the sorted zipped colors for x, y values

        colors = colors_val = colors_index = colors_zipped = [None] * 3

        # loop 3 times to caluclate the Blue, Green, Red
        for i in range(3):
            colors[i] = cv2.calcHist([img], [i], None, [256], [0, 256])
            colors_val[i] = colors[i].ravel()
            colors_zipped[i] = zip(colors_val[i], indices)
            colors_sorted = sorted(colors_zipped[i], reverse=True)
            colors_index[i] = [(x, y) for y, x in colors_sorted]

        # index of highest peak in histogram for the blue, green red
        blue_index = colors_index[0][0][0] # [0][0] to get the blue and [0] to get the x_axis
        green_index = colors_index[1][1][0] # [1][1] to get the green and [0] to get the x_axis
        red_index = colors_index[2][2][0] # [2][2] to get the red and [0] to get the x_axis

        # return the highest index
        return maximum(blue_index, green_index, red_index)

    # return the highest_spike value found
    return find_highest_spike()

# store the highest x_axis value
highest_x_axis_value = find_image_algorithm(img)

# used to set the average points of color
points_of_color = 110
fixed_img = None

# run the faded algorithm if the highest_x_axis_value is greater than the points_of_color
if (highest_x_axis_value > points_of_color):
    fixed_img = damaged(img)
    plt.imshow(fixed_img, cmap='gray')
else:
    fixed_img = faded(img)
    fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2RGB)
    plt.imshow(fixed_img)

plt.title('Fixed Image')
plt.show()