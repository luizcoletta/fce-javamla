import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class ObtainCoAssMatrix {

	int labelMatrix[][] = null;
	int numPositions = 1500;
	int numObj = 0;
	int numClusterers = 2;

	public ObtainCoAssMatrix(){

		try {

			labelMatrix = new int [numPositions][numClusterers];
			for (int i=0; i<numPositions; i++){
				for (int j=0; j<numClusterers; j++){
					labelMatrix[i][j] = -1;
				}
			}

			obtain();

			for (int i=0; i<numPositions; i++){
				if (labelMatrix[i][1] == -1){
					numObj = i;
					break;
				}
			}

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private void obtain() throws FileNotFoundException, IOException{

		for (int i=1; i<=numClusterers; i++){
			load(String.valueOf(i));
		}
	}

	private void load(String arquivo) throws FileNotFoundException, IOException {

		int clusters = new File("clustering/" + arquivo).list().length-2;
		
		for (int i=0; i<clusters; i++){

			String nFile = "clustering/" + arquivo + "/cluster_" + String.valueOf(i) + ".txt";

			File file = new File(nFile);
			if (file.exists()) {

				BufferedReader br = new BufferedReader(new FileReader(nFile));
				StringBuffer bufSaida = new StringBuffer();

				String linha;
				while((linha = br.readLine()) != null ){
					if (linha.substring(0, 2).equals("ID")){

						int ind = Integer.valueOf(linha.substring(3, linha.indexOf(' ',3)));
						labelMatrix[ind-1][Integer.valueOf(arquivo)-1] = i;
					}	
				}
			}
		}
	}

	public float[][] createCMatrix() {

		float caMatrix[][] = new float [numObj][numObj];

		for (int i = 0; i < numClusterers; i++) {
			for (int j = 0; j < numObj; j++) {
				for (int k = 0; k < numObj; k++) {
					if (labelMatrix[j][i] == labelMatrix[k][i]){
						caMatrix[j][k] += 1;
					}
					if (i==(numClusterers-1)){
						caMatrix[j][k] = caMatrix[j][k]/numClusterers;
					}
				}
			}
		}
		return caMatrix;
	}
}
