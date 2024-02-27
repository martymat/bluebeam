import cv2

# Read the image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Define the symbol template (assuming a binary image for simplicity)
symbol_template = cv2.imread('symbol_template.jpg', cv2.IMREAD_GRAYSCALE)

# Perform template matching
result = cv2.matchTemplate(image, symbol_template, cv2.TM_CCOEFF_NORMED)

# Define a threshold to filter out weak matches
threshold = 0.8
locations = np.where(result >= threshold)

# Count the number of occurrences
symbol_count = len(locations[0])

print("Number of symbol occurrences:", symbol_count)