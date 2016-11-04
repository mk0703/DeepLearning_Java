package utill;

public final class ActivatonFunctions {
	public static int step(double preActivation) {
		if (preActivation >= 0) {
			return 1;
		}
		else {
			return -1;
		}
	}


}
