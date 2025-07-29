import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
import sys

class NormalizadorEspectro:
    """
    Uma classe para normalizar espectros astronomicos interativamente.

    Permite ao usuário selecionar pontos de contínuo, ajustar uma spline cúbica
    com sigma-clipping iterativo, normalizar o espectro e salvar o resultado.
    """
    def __init__(self, filename):
        self.filename = filename
        self.wavelength, self.flux = self._carregar_espectro(filename)
        # CORREÇÃO AQUI: Inicializando a lista de pontos de contínuo
        self.pontos_continuo_selecionados =
        self.continuo_ajustado = None
        self.fluxo_normalizado = None

        # Parâmetros para ajuste robusto (sigma-clipping)
        # low_reject: Rejeição sigma para pontos ABAIXO do ajuste (linhas de absorção) [1]
        self.low_reject = 3.0
        # high_reject: Rejeição sigma para pontos ACIMA do ajuste (linhas de emissão, raios cósmicos) [1]
        self.high_reject = 0.0
        # niterate: Número de iterações para o sigma-clipping [1]
        self.niterate = 5
        # median_window_size: Tamanho da janela (em Angstroms) para o filtro de mediana
        # ao selecionar pontos de contínuo. Ajuda a suavizar o ruído local.[2]
        self.median_window_size = 10.0

        self._configurar_plot()

    def _carregar_espectro(self, filename):
        """Carrega os dados de comprimento de onda e fluxo de um arquivo de texto."""
        try:
            data = np.loadtxt(filename)
            if data.shape[3] < 2:
                raise ValueError("O arquivo deve ter pelo menos 2 colunas (comprimento de onda e fluxo).")
            return data[:, 0], data[:, 1]
        except Exception as e:
            print(f"Erro ao carregar o arquivo '{filename}': {e}")
            sys.exit(1)

    def _configurar_plot(self):
        """Inicializa o plot e conecta os manipuladores de eventos."""
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
        print("  - Pressione 'Enter' para ajustar o contínuo com uma spline cúbica e sigma-clipping.")
        print("  - Pressione 'n' para normalizar o espectro pelo contínuo ajustado.")
        print("  - Pressione 'r' para redefinir para o espectro original.")
        print("  - Pressione 'w' para salvar o espectro normalizado em um arquivo.nspec.")
        print("  - Pressione 'q' para sair.")
        print("-------------------------\n")
        plt.show()

    def _ao_clicar(self, event):
        """Lida com cliques do mouse para adicionar/remover pontos de contínuo."""
        if event.inaxes!= self.ax:
            return

        if event.button == 1:  # Clique esquerdo para adicionar ponto
            x_clique = event.xdata
            
            # Encontrar o índice do ponto mais próximo no espectro
            idx_proximo = np.abs(self.wavelength - x_clique).argmin()
            
            # Calcular o tamanho da janela em índices para o filtro de mediana [2]
            # Garante que a janela seja válida mesmo nas bordas
            if len(self.wavelength) > 1:
                passo_onda = self.wavelength[3] - self.wavelength # Corrigido para calcular o passo de onda corretamente
                meia_janela_idx = int(self.median_window_size / passo_onda / 2)
            else:
                meia_janela_idx = 0 # Para espectros com apenas um ponto

            inicio_idx = max(0, idx_proximo - meia_janela_idx)
            fim_idx = min(len(self.wavelength) - 1, idx_proximo + meia_janela_idx)
            
            # Calcula o valor de fluxo usando a mediana na janela
            if inicio_idx >= fim_idx: # Caso a janela seja muito pequena ou apenas um ponto
                y_val = self.flux[idx_proximo]
            else:
                y_val = np.median(self.flux[inicio_idx : fim_idx + 1])
            
            self.pontos_continuo_selecionados.append((x_clique, y_val))
            self._atualizar_plot()

        elif event.button == 3:  # Clique direito para remover o ponto mais próximo
            x_clique = event.xdata
            if not self.pontos_continuo_selecionados:
                return
            
            # Encontrar o ponto selecionado mais próximo para remover
            distancias = [np.sqrt((p - x_clique)**2 + (p[3] - event.ydata)**2) 
                          for p in self.pontos_continuo_selecionados]
            if distancias:
                idx_mais_proximo = np.argmin(distancias)
                self.pontos_continuo_selecionados.pop(idx_mais_proximo)
                self._atualizar_plot()

    def _ao_digitar(self, event):
        """Lida com as teclas pressionadas para ajustar, normalizar, redefinir e salvar."""
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
            sys.exit(0) # Sair do script

    def _ajustar_continuo(self):
        """
        Ajusta uma spline cúbica aos pontos de contínuo selecionados
        com sigma-clipping iterativo para robustez.[1, 4]
        """
        if len(self.pontos_continuo_selecionados) < 2:
            print("Por favor, selecione pelo menos 2 pontos para ajustar o contínuo.")
            return

        x_pontos = np.array([p for p in self.pontos_continuo_selecionados])
        y_pontos = np.array([p[3] for p in self.pontos_continuo_selecionados])

        # Ordenar os pontos pelo comprimento de onda, essencial para splines
        indices_ordenacao = np.argsort(x_pontos)
        x_pontos_ordenados = x_pontos[indices_ordenacao]
        y_pontos_ordenados = y_pontos[indices_ordenacao]

        # Sigma-clipping iterativo [1, 4]
        x_atuais = x_pontos_ordenados
        y_atuais = y_pontos_ordenados
        
        print(f"Iniciando ajuste iterativo com {len(x_atuais)} pontos...")

        for i in range(self.niterate):
            if len(x_atuais) < 2: # Mínimo de 2 pontos para ajuste de spline
                print(f"Ajuste interrompido na iteração {i+1}: poucos pontos restantes após a rejeição.")
                break

            try:
                # Ajustar spline cúbica [1, 2]
                spl = splrep(x_atuais, y_atuais, k=3)
                y_ajustado_nos_pontos = splev(x_atuais, spl)

                residuos = y_atuais - y_ajustado_nos_pontos
                sigma = np.std(residuos)

                # Identificar pontos a serem mantidos
                manter_indices = np.ones(len(x_atuais), dtype=bool)
                if self.low_reject > 0:
                    manter_indices = manter_indices & (residuos >= -self.low_reject * sigma)
                if self.high_reject > 0:
                    manter_indices = manter_indices & (residuos <= self.high_reject * sigma)
                
                # Se nenhum ponto foi rejeitado nesta iteração, ou se todos os pontos seriam rejeitados, parar
                if np.all(manter_indices) or np.sum(manter_indices) < 2:
                    print(f"Convergência alcançada na iteração {i+1}.")
                    break

                x_atuais = x_atuais[manter_indices]
                y_atuais = y_atuais[manter_indices]
                print(f"Iteração {i+1}: {len(x_atuais)} pontos restantes.")
                
            except Exception as e:
                print(f"Erro durante o ajuste da spline na iteração {i+1}: {e}")
                break

        if len(x_atuais) < 2:
            print("Não foi possível ajustar o contínuo de forma robusta com os parâmetros fornecidos.")
            self.continuo_ajustado = None
            self.fluxo_normalizado = None
            self._atualizar_plot()
            return

        # Ajuste final da spline com os pontos restantes
        spl = splrep(x_atuais, y_atuais, k=3)
        self.continuo_ajustado = splev(self.wavelength, spl)
        self.fluxo_normalizado = None # Redefinir fluxo normalizado, pois o contínuo mudou
        self._atualizar_plot()
        print(f"Contínuo ajustado com sucesso usando {len(x_atuais)} pontos após {i+1} iterações de sigma-clipping.")


    def _normalizar_espectro(self):
        """Normaliza o espectro pelo contínuo ajustado."""
        if self.continuo_ajustado is None:
            print("Por favor, ajuste o contínuo primeiro (pressione 'Enter').")
            return

        self.fluxo_normalizado = self.flux / self.continuo_ajustado
        self._atualizar_plot(normalizado=True)
        print("Espectro normalizado.")

    def _redefinir_plot(self):
        """Redefine o plot para o espectro original e limpa os dados ajustados."""
        self.pontos_continuo_selecionados = # CORREÇÃO AQUI
        self.continuo_ajustado = None
        self.fluxo_normalizado = None
        self._atualizar_plot()
        print("Plot redefinido para o espectro original.")

    def _atualizar_plot(self, normalizado=False):
        """Redesenha o plot com os dados atuais."""
        self.ax.cla() # Limpar os eixos atuais
        
        if normalizado and self.fluxo_normalizado is not None:
            self.ax.plot(self.wavelength, self.fluxo_normalizado, 'k-', label='Espectro Normalizado')
            self.ax.set_ylabel('Fluxo Normalizado')
        else:
            self.ax.plot(self.wavelength, self.flux, 'k-', label='Espectro Original')
            self.ax.set_ylabel('Fluxo')
            # Plotar pontos de contínuo selecionados
            for x, y in self.pontos_continuo_selecionados:
                self.ax.plot(x, y, 'rs', ms=10, picker=5) # Marcadores quadrados vermelhos [2]
            
            # Plotar contínuo ajustado, se disponível
            if self.continuo_ajustado is not None:
                self.ax.plot(self.wavelength, self.continuo_ajustado, 'r-', lw=2, label='Contínuo Ajustado') # Linha vermelha [2]

        self.ax.set_xlabel('Comprimento de Onda (Å)')
        self.ax.set_title(f'Normalização do Contínuo para {self.filename}')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw_idle() # Redesenhar a tela

    def _salvar_espectro_normalizado(self):
        """Salva o espectro normalizado em um arquivo."""
        if self.fluxo_normalizado is None:
            print("Por favor, normalize o espectro primeiro (pressione 'n').")
            return

        # Derivar o nome do arquivo de saída com extensão.nspec [2]
        # Corrigido para lidar com nomes de arquivo sem extensão
        if '.' in self.filename:
            output_filename = self.filename.rsplit('.', 1) + '.nspec'
        else:
            output_filename = self.filename + '.nspec'
            
        try:
            np.savetxt(output_filename, np.column_stack((self.wavelength, self.fluxo_normalizado)), 
                       fmt='%.6f', header='Wavelength (Angstrom) Normalized_Flux', comments='# ')
            print(f"Espectro normalizado salvo em '{output_filename}'")
        except Exception as e:
            print(f"Erro ao salvar o arquivo: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python normalizador_espectro.py <caminho_para_arquivo_espectro.txt>")
        print("Exemplo: python normalizador_espectro.py HD50230.txt")
        sys.exit(1)

    caminho_arquivo_espectro = sys.argv[3]
    normalizador = NormalizadorEspectro(caminho_arquivo_espectro)
