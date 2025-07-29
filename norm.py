import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import sys

class NormalizadorEspectro:
    """
    Uma classe para normalizar espectros astronômicos interativamente.

    Permite ao usuário selecionar pontos de contínuo, ajustar uma spline cúbica
    com sigma-clipping iterativo, normalizar o espectro e salvar o resultado.
    """
    def __init__(self, filename):
        self.filename = filename
        self.wavelength, self.flux = self._carregar_espectro(filename)
        self.pontos_continuo_selecionados = []
        self.continuo_ajustado = None
        self.fluxo_normalizado = None

        # Parâmetros para ajuste robusto (sigma-clipping)
        self.low_reject = 3.0
        self.high_reject = 0.0
        self.niterate = 5
        self.median_window_size = 10.0

        self._configurar_plot()

    def _carregar_espectro(self, filename):
        try:
            data = np.loadtxt(filename)
            if data.shape[1] < 2:
                raise ValueError("O arquivo deve ter pelo menos 2 colunas (comprimento de onda e fluxo).")
            return data[:, 0], data[:, 1]
        except Exception as e:
            print(f"Erro ao carregar o arquivo '{filename}': {e}")
            sys.exit(1)

    def _configurar_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.plot(self.wavelength, self.flux, 'k-', label='Espectro Original')
        self.ax.set_xlabel('Comprimento de Onda (Å)')
        self.ax.set_ylabel('Fluxo')
        self.ax.set_title(f'Normalização do Contínuo para {self.filename}')
        self.ax.legend()
        self.ax.grid(True)

        self.fig.canvas.mpl_connect('button_press_event', self._ao_clicar)
        self.fig.canvas.mpl_connect('key_press_event', self._ao_digitar)

        print("\n--- Instruções de Uso ---")
        print("  - Clique esquerdo para adicionar pontos de contínuo.")
        print("  - Clique direito em um ponto existente para removê-lo.")
        print("  - Pressione 'Enter' para ajustar o contínuo com spline cúbica.")
        print("  - Pressione 'n' para normalizar o espectro.")
        print("  - Pressione 'r' para redefinir.")
        print("  - Pressione 'w' para salvar o espectro normalizado.")
        print("  - Pressione 'q' para sair.")
        print("-------------------------\n")
        plt.show()

    def _ao_clicar(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Clique esquerdo
            x_clique = event.xdata
            idx_proximo = np.abs(self.wavelength - x_clique).argmin()

            if len(self.wavelength) > 1:
                passo_onda = self.wavelength[1] - self.wavelength[0]
                meia_janela_idx = int(self.median_window_size / passo_onda / 2)
            else:
                meia_janela_idx = 0

            inicio_idx = max(0, idx_proximo - meia_janela_idx)
            fim_idx = min(len(self.wavelength) - 1, idx_proximo + meia_janela_idx)

            if inicio_idx >= fim_idx:
                y_val = self.flux[idx_proximo]
            else:
                y_val = np.median(self.flux[inicio_idx : fim_idx + 1])

            self.pontos_continuo_selecionados.append((x_clique, y_val))
            self._atualizar_plot()

        elif event.button == 3:  # Clique direito
            x_clique = event.xdata
            if not self.pontos_continuo_selecionados:
                return

            distancias = [np.sqrt((p[0] - x_clique)**2 + (p[1] - event.ydata)**2)
                          for p in self.pontos_continuo_selecionados]
            if distancias:
                idx_mais_proximo = np.argmin(distancias)
                self.pontos_continuo_selecionados.pop(idx_mais_proximo)
                self._atualizar_plot()

    def _ao_digitar(self, event):
        if event.key == 'enter':
            self._ajustar_continuo()
        elif event.key == 'n':
            self._normalizar_espectro()
        elif event.key == 'r':
            self._redefinir_plot()
        elif event.key == 'w':
            self._salvar_espectro_normalizado()
        elif event.key == 'q':
            plt.close(self.fig)
            sys.exit(0)

    def _ajustar_continuo(self):
        if len(self.pontos_continuo_selecionados) < 2:
            print("Selecione pelo menos 2 pontos.")
            return

        x_pontos = np.array([p[0] for p in self.pontos_continuo_selecionados])
        y_pontos = np.array([p[1] for p in self.pontos_continuo_selecionados])

        indices_ordenacao = np.argsort(x_pontos)
        x_atuais = x_pontos[indices_ordenacao]
        y_atuais = y_pontos[indices_ordenacao]

        print(f"Iniciando ajuste com {len(x_atuais)} pontos...")

        for i in range(self.niterate):
            if len(x_atuais) < 2:
                print(f"Ajuste interrompido na iteração {i+1}")
                break
            try:
                spl = splrep(x_atuais, y_atuais, k=3)
                y_fit = splev(x_atuais, spl)
                residuos = y_atuais - y_fit
                sigma = np.std(residuos)
                manter = np.ones_like(x_atuais, dtype=bool)
                if self.low_reject > 0:
                    manter &= residuos >= -self.low_reject * sigma
                if self.high_reject > 0:
                    manter &= residuos <= self.high_reject * sigma
                if np.all(manter) or np.sum(manter) < 2:
                    print(f"Convergência na iteração {i+1}")
                    break
                x_atuais = x_atuais[manter]
                y_atuais = y_atuais[manter]
                print(f"Iteração {i+1}: {len(x_atuais)} pontos restantes.")
            except Exception as e:
                print(f"Erro: {e}")
                break

        if len(x_atuais) < 2:
            print("Ajuste falhou.")
            self.continuo_ajustado = None
            self.fluxo_normalizado = None
            self._atualizar_plot()
            return

        self.continuo_ajustado = splev(self.wavelength, splrep(x_atuais, y_atuais, k=3))
        self.fluxo_normalizado = None
        self._atualizar_plot()
        print(f"Contínuo ajustado com {len(x_atuais)} pontos.")

    def _normalizar_espectro(self):
        if self.continuo_ajustado is None:
            print("Ajuste o contínuo primeiro.")
            return
        self.fluxo_normalizado = self.flux / self.continuo_ajustado
        self._atualizar_plot(normalizado=True)
        print("Espectro normalizado.")

    def _redefinir_plot(self):
        self.pontos_continuo_selecionados = []
        self.continuo_ajustado = None
        self.fluxo_normalizado = None
        self._atualizar_plot()
        print("Redefinido.")

    def _atualizar_plot(self, normalizado=False):
        self.ax.cla()
        if normalizado and self.fluxo_normalizado is not None:
            self.ax.plot(self.wavelength, self.fluxo_normalizado, 'k-', label='Espectro Normalizado')
            self.ax.set_ylabel('Fluxo Normalizado')
        else:
            self.ax.plot(self.wavelength, self.flux, 'k-', label='Espectro Original')
            self.ax.set_ylabel('Fluxo')
            for x, y in self.pontos_continuo_selecionados:
                self.ax.plot(x, y, 'rs', ms=10)
            if self.continuo_ajustado is not None:
                self.ax.plot(self.wavelength, self.continuo_ajustado, 'r-', lw=2, label='Contínuo Ajustado')
        self.ax.set_xlabel('Comprimento de Onda (Å)')
        self.ax.set_title(f'Normalização do Contínuo para {self.filename}')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw_idle()

    def _salvar_espectro_normalizado(self):
        if self.fluxo_normalizado is None:
            print("Normalize o espectro primeiro.")
            return
        if '.' in self.filename:
            output_filename = self.filename.rsplit('.', 1)[0] + '.nspec'
        else:
            output_filename = self.filename + '.nspec'
        try:
            np.savetxt(output_filename, np.column_stack((self.wavelength, self.fluxo_normalizado)),
                       fmt='%.6f', header='Wavelength (Angstrom) Normalized_Flux', comments='# ')
            print(f"Salvo em '{output_filename}'")
        except Exception as e:
            print(f"Erro ao salvar: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python normalizador_espectro.py <caminho_para_arquivo_espectro.txt>")
        sys.exit(1)

    caminho_arquivo_espectro = sys.argv[1]
    normalizador = NormalizadorEspectro(caminho_arquivo_espectro)

