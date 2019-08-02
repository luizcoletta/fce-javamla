import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class ARFFToMatlab {

	public static void salvar(String arquivo, StringBuffer conteudo, boolean adicionar)
	throws IOException {

		FileWriter fw = new FileWriter(arquivo, adicionar);
		fw.write(conteudo.toString());
		fw.close();
	}

	public static StringBuffer carregar(String arquivo)
	throws FileNotFoundException, IOException {

		File file = new File(arquivo);

		if (!file.exists()) {
			return null;
		}

		BufferedReader br = new BufferedReader(new FileReader(arquivo));
		StringBuffer bufSaida = new StringBuffer();

		String linha;
		boolean cabecalho = true;
		while((linha = br.readLine()) != null ){
			if (!cabecalho){
				linha = linha.replace(',', '\t');
				bufSaida.append(linha + "\r\n");
			}	
			if (cabecalho){
				if (linha.compareToIgnoreCase("@data")==0)
					cabecalho = false;
			}
		}
		br.close();
		return bufSaida;
	}
}
