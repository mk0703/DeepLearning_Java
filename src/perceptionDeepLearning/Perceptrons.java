package perceptionDeepLearning;

import java.util.Random;

import utill.ActivatonFunctions;
import utill.GaussianDistribution;

public class Perceptrons {

	public int nIn;
	public double [] w;

	public Perceptrons(int nIn) {
		this.nIn = nIn;
		w = new double [nIn];
	}

	public int train(double [] x , int t , double rate) {
		int classified = 0;
		double c = 0.;

		//データが正しく分類されてるかチェック
		for(int i =0 ; i< nIn ; i++){
			c += w[i] * x[i] * t;
		}

		if( c > 0) {
			classified = 1;
		}
		else {
			for(int i = 0 ; i < nIn ; i++) {
				w[i] += rate * x[i] * t;
			}
		}

		return classified;
	}

	public int predict (double [] x) {
		double preActivation = 0;

		for(int i =0 ; i < nIn ; i++) {
			preActivation += w[i] * x[i];
		}

		//utill内のstep関数を使用
		return ActivatonFunctions.step(preActivation);
	}

	public static void main(String[] args) {

		final Random rng = new Random(1234);

		final int train_N = 1000;//トレーニングデータ数
		final int test_N = 200;//テストデータ数
		final int nIn = 2;//ニューロン数

		double[][] train_X = new double[train_N][nIn];//トレーニングデータ
		int[] train_T = new int[train_N];//データラベル

		double[][] test_X = new double[test_N][nIn];//テストデータ
		int[] test_T = new int[test_N];//データラベル
		int[] predicted_T = new int[test_N];//予測結果

		final int epochs = 2000;//トレーニングの上限数
		final double rate = 1.0;//学習率

		//サンプルデータの定義(正規分布)
		GaussianDistribution g1 = new GaussianDistribution(-2.0 , 1.0 , rng);
		GaussianDistribution g2 = new GaussianDistribution(2.0 , 1.0 , rng);

		for(int i = 0; i < train_N/2 - 1; i++){
			train_X[i][0] = g1.random();
			train_X[i][1] = g2.random();
			train_T[i] = 1;
		}

		for(int i = 0; i < test_N/2 - 1; i++) {
			test_X[i][0] = g1.random();
			test_X[i][1] = g2.random();
			test_T[i] = 1;
		}

		for(int i = train_N/2 ; i < train_N; i++) {
			train_X[i][0] = g2.random();
			train_X[i][1] = g1.random();
			train_T[i] = -1;
		}

		for( int i = test_N/2; i < test_N; i++) {
			test_X[i][0] = g2.random();
			test_X[i][1] = g1.random();
			test_T[i] = -1;
		}


		int epoch = 0;

		Perceptrons classifier = new Perceptrons(nIn);

		while(true) {
			int classified = 0;

			for(int i = 0; i < train_N; i++) {
				classified += classifier.train(train_X[i], train_T[i], rate);
			}

			//全データが分類されたら終了
			if(classified == train_N) {
				break;
			}

			epoch += 1;

			if(epoch > epochs) {
				break;
			}
		}

		for(int i = 0; i < test_N; i++) {
			//分類結果を格納
			predicted_T[i] = classifier.predict(test_X[i]);
		}

		int [][] confusionMatrix = new int[2][2];
		double accuracy = 0;
		double precision = 0;
		double recall = 0;

		for(int i = 0; i < test_N; i++) {
			if(predicted_T[i] > 0) {
				if(test_T[i] >0) {
					accuracy += 1;
					precision += 1;
					recall += 1;
					confusionMatrix[0][0] += 1;
				}

				else {
					confusionMatrix[1][0] += 1;
				}
			}

			else {

				if(test_T[i] > 0) {
					confusionMatrix[0][1] += 1;
				}

				else {
					accuracy += 1;
					confusionMatrix[1][1] += 1;
				}
			}
		}

		//各確率を計算
		accuracy /= test_N;
		precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
		recall /= confusionMatrix[0][0] + confusionMatrix[0][1];

		//結果の表示
		System.out.println("+++++Perceptrons Evaluation+++++");
		System.out.printf("Accuracy(正解率):   %.1f %%\n" , accuracy * 100);
		System.out.printf("Precision(精度):    %.1f %%\n" , precision * 100);
		System.out.printf("Recall(再現率):     %.1f %%\n" , recall * 100);
	}

}
