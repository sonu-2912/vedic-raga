import scripts.colab_common as common
from scripts.compact_residual_fusdense import model_from
common.model_from=model_from
from scripts.colab_evaluate import main
if __name__=='__main__':main()
