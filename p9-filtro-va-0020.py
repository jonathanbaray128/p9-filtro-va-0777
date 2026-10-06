import cv2
import numpy as np

# Cargar imagen
imagen = cv2.imread("rinoceronte.jpg")

if imagen is None:
    print("No se encontró la imagen rinoceronte.jpg")
else:
    print("Imagen cargada correctamente.")

    # Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Filtro Canny
    bordes_canny = cv2.Canny(gris, 100, 200)

    # Filtro Sobel
    sobel_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)

    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # Combinar Sobel X y Sobel Y
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Filtro Laplaciano
    laplaciano = cv2.Laplacian(gris, cv2.CV_64F)
    laplaciano = cv2.convertScaleAbs(laplaciano)

    # Mostrar resultados
    cv2.imshow("Imagen original", imagen)
    cv2.imshow("Escala de grises", gris)
    cv2.imshow("Bordes Canny", bordes_canny)
    cv2.imshow("Filtro Sobel", sobel)
    cv2.imshow("Filtro Laplaciano", laplaciano)

    # Guardar resultados
    cv2.imwrite("bordes_canny.jpg", bordes_canny)
    cv2.imwrite("filtro_sobel.jpg", sobel)
    cv2.imwrite("filtro_laplaciano.jpg", laplaciano)

    print("Filtros aplicados correctamente.")
    print("Presiona una tecla para cerrar.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()