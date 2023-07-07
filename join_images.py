import cv2
import numpy as np

# Function that can joins together multiple images of same height and width by specifying no of columns.
def Join_Images(
    image_list, n_cols=2, window_name="Joined_Window", scale=20, create_window=True
):
    """
    ### Input:
    * image_list: List of images of same height and width (even of different channels).
    * n_cols: Number of columns of images in final joined image.
    * window_name: Window name.
    * scale: Scale of image.
    * create_window: If `True` then window will be created else window won't be created.
    ### Output:
    ##### If create_window is `True` then:
    Window that has all the images joined and its matrix with height and width
    ##### else:
    Joined image matrix with height and width.
    """

    n_images = len(image_list)
    img_height, img_width = image_list[0].shape[0], image_list[0].shape[1]
    n_rows = int(np.ceil(n_images / n_cols))
    n_images_window = n_rows * n_cols

    # Getting window height and width and scaling it.
    window_height = img_height // n_cols
    window_width = img_width // n_rows
    window_height, window_width = window_height * scale, window_width * scale

    if create_window:
        _ = cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width=window_width, height=window_height)

    # Checking if enough images are there to fill the window else generate black full black images.
    if n_images != n_images_window:
        blank_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    new_image = []
    for i in range(n_cols):
        img = image_list[i]
        # Converting grayscale image to BGR format.
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        new_image.append(img)
    new_image = np.hstack(new_image)

    for i in range(1, n_rows):
        # Creating each row of images
        new_row = []
        for j in range(n_cols):
            # Checking if the number of images required to fill the window exceeds the number of images given as input.
            if n_images <= i * n_cols + j:
                # Appending blank image as replacement for image that doesn't exist.
                new_row.append(blank_image)
            else:
                img = image_list[i * n_cols + j]

                # Converting grayscale image to BGR format.
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                new_row.append(img)
        new_row = np.hstack(new_row)

        # Stacking rows of images into new image.
        new_image = np.vstack([new_image, new_row])

    # Reshaping the image into the new window size.
    new_image = cv2.resize(new_image, (window_height, window_width))

    if create_window:
        cv2.imshow(window_name, new_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    return new_image, window_height, window_width
