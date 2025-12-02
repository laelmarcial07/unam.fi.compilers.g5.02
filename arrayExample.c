#include <stdio.h>

int main() {
    int A[5] = {1, 2, 3, 4, 5};
    int B[5] = {10, 20, 30, 40, 50};
    int C[5];

    // Suma elemento por elemento
    for (int i = 0; i < 5; i++) {
        C[i] = A[i] + B[i];
    }

    // Imprimir el resultado
    printf("Resultado de la suma de arreglos:\n");
    for (int i = 0; i < 5; i++) {
        printf("%d ", C[i]);
    }

    return 0;
}