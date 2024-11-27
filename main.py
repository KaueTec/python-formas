import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


class FormDetector:
    def load_image(self, path_image):
        image = cv2.imread(path_image)
        if image is None:
            raise ValueError(f"Imagem não encontrada: {path_image}")
        return image

    def convert_to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def edge_detection(self, image):
        return cv2.Canny(image, 50, 150)

    def find_lines(self, imagem_bordas):
        contornos, _ = cv2.findContours(
            imagem_bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contornos

    def classify_forms(self, contornos, imagem_original):
        formas = {
            "triangulos": 0,
            "quadrados": 0,
            "circulos": 0,
            "pentagono": 0,
            "outros": 0
        }

        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            aproximacao = cv2.approxPolyDP(contorno, 0.04 * perimetro, True)

            if len(aproximacao) == 3:
                formas["triangulos"] += 1
                cv2.drawContours(imagem_original, [contorno], 0, (0, 255, 0), 2)

            elif len(aproximacao) == 4:
                x, y, w, h = cv2.boundingRect(contorno)
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    formas["quadrados"] += 1
                    cv2.drawContours(imagem_original, [contorno], 0, (0, 255, 0), 2)
                else:
                    formas["outros"] += 1

            elif len(aproximacao) == 5:
                formas["pentagono"] += 1
                cv2.drawContours(imagem_original, [contorno], 0, (0, 255, 0), 2)

            elif len(aproximacao) > 5:
                area = cv2.contourArea(contorno)
                perimetro_quadrado = perimetro * perimetro
                circularity = 4 * np.pi * area / perimetro_quadrado

                if circularity > 0.8:
                    formas["circulos"] += 1
                    cv2.drawContours(imagem_original, [contorno], 0, (0, 0, 255), 2)
                else:
                    formas["outros"] += 1
        return formas

    def view_results(self, imagem_original, formas):
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB))
        plt.title("Formas geométricas detectadas")
        plt.axis("off")

        legenda = \
            f"""Formas detectadas:
                Triângulos: {formas["triangulos"]}
                Quadrados: {formas["quadrados"]}
                Círculos: {formas["circulos"]}
                Pentágono: {formas["pentagono"]}
                Outras formas: {formas["outros"]}"""

        plt.text(imagem_original.shape[1] // 10,
                 imagem_original.shape[0] - (imagem_original.shape[0] // 10),
                 legenda,
                 fontsize=10,
                 verticalalignment="bottom",
                 horizontalalignment="left",
                 bbox=dict(facecolor='white', alpha=0.5))

        plt.tight_layout()
        plt.show()


def main():
    # Função para abrir o diálogo de seleção de arquivo
    def open_file():
        caminho_imagem = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )

        if caminho_imagem:
            detector = FormDetector()
            try:
                imagem = detector.load_image(caminho_imagem)
                imagem_cinza = detector.convert_to_grayscale(imagem)
                imagem_blur = detector.apply_blur(imagem_cinza)
                imagem_bordas = detector.edge_detection(imagem_blur)
                contornos = detector.find_lines(imagem_bordas)
                formas = detector.classify_forms(contornos, imagem)
                detector.view_results(imagem, formas)
            except Exception as e:
                print(f"Erro no processamento de imagem: {e}")

    # Criar a interface gráfica
    root = tk.Tk()
    root.title("Detecção de Formas Geométricas")

    btn_open = tk.Button(root, text="Importar Imagem", command=open_file)
    btn_open.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    main()