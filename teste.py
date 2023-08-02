from PIL import Image
import os

def convert_webp_to_png(input_path, output_path):
    try:
        # Abre a imagem WebP
        with Image.open(input_path) as img:
            # Salva a imagem como PNG
            img.save(output_path, "PNG")
        
        os.remove(input_path)
            
            
    except Exception as e:
        print(f"Erro ao converter a imagem: {e}")

def convert_images_in_folder(folder_path):
    # Lista todos os arquivos na pasta
    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file.lower().endswith('.webp'):
            # Se o arquivo for uma imagem WebP, converte para PNG
            png_filename = file.replace('.webp', '.png')
            output_path = os.path.join(folder_path, png_filename)
            convert_webp_to_png(file_path, output_path)
        
def main():
    # Caminho da pasta pai "Imagens"
    parent_folder = "Imagens"

    # Subpastas a serem percorridas
    subfolders = ["Ass", "Boobs", "Pussy"]

    for subfolder in subfolders:
        folder_path = os.path.join(parent_folder, subfolder)
        if os.path.exists(folder_path):
            print(f"Convertendo imagens na pasta: {folder_path}")
            convert_images_in_folder(folder_path)
        else:
            print(f"A pasta {subfolder} não existe em {parent_folder}.")

if __name__ == "__main__":
    main()
