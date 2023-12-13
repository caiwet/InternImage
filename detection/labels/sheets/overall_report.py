from PIL import Image
import os

# List of folders containing the images
hospitals = ['Ascension-Seton', 'Cedars-Sinai','Chiang_Mai_University', 
            'Fundación_Santa_Fe_de_Bogotá', 'Lawson_Health', 
            'Morales_Meseguer_Hospital', 'National_University_of_Singapore',
            'Newark_Beth_Israel_Medical_Center', 'NYU_Langone_Health',
            'Osaka_City_University', 'Rhode_Island_Hospital', 
            'Sunnybrook_Research_Institute', 'Technical_University_of_Munich',
            'Universitätsklinikum_Essen', 'Universitätsklinikum_Tübingen', 
            'University_of_Miami']

def create_combined_image(image_name, outfile_name):
    # Determine the number of rows and columns to create a square image
    num_images = len(hospitals)
    num_cols = int(num_images ** 0.5)
    num_rows = (num_images + num_cols - 1) // num_cols

    # Load the first image to get its size
    first_image_path = os.path.join(hospitals[0], image_name)
    if os.path.exists(first_image_path):
        first_image = Image.open(first_image_path)
        width, height = first_image.size

        # Calculate the size of the square canvas
        canvas_width = num_cols * width
        canvas_height = num_rows * height

        # Create a new image with the square canvas size
        combined_image = Image.new("RGB", (canvas_width, canvas_height))

        # Initialize variables to keep track of the current row and column
        current_row = 0
        current_col = 0

        # Paste the images onto the square canvas
        for folder in hospitals:
            image_path = os.path.join(folder, image_name)
            if os.path.exists(image_path):
                image = Image.open(image_path)
                combined_image.paste(image, (current_col * width, current_row * height))
                current_col += 1
                if current_col == num_cols:
                    current_col = 0
                    current_row += 1

        # Save the combined square image
        combined_image.save(f"results/{outfile_name}")
        # Close all the images to free up resources
        first_image.close()
    else:
        print("No images found in the specified folders.")

if __name__ == "__main__":
    create_combined_image(image_name = 'y_err.png', outfile_name = 'combined_y_err.png')
    create_combined_image(image_name = 'cum_err.png', outfile_name = 'combined_cum_err.png')
    create_combined_image(image_name = 'histogram.png', outfile_name = 'combined_euc_err.png')
