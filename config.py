class Options():
    def __init__(self):
        super().__init__()

        self.Input_Path_Test = 'E://RSCityScape_small/test/input/'
        self.Target_Path_Test = 'E://RSCityScape_small/test/target/'
        self.Result_Path_Test = 'E://RSCityScape_small/test/result_Restormer/'
        self.MODEL_PATH = './model_best.pth'
        self.Num_Works = 4
        self.CUDA_USE = True