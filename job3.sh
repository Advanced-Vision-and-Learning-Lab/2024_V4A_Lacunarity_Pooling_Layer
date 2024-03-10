        if aggFunc == "local":
            if poolingLayer == "max":
                m.global_pool = nn.MaxPool2d(kernel_size=(kernel, kernel), stride =(stride, stride), padding=(padding, padding))
            elif poolingLayer == "avg":                                                                                                                                                                                                                            
                m.global_pool = nn.global_pool2d(kernel_size=(kernel, kernel), stride =(stride, stride), padding=(padding, padding))
            elif poolingLayer == "Base_Lacunarity":
                m.global_pool = Base_Lacunarity(model_name=model_name, scales=scales, kernel=(kernel, kernel), stride =(stride, stride), bias=bias)
            elif poolingLayer == "Pixel_Lacunarity":
                m.global_pool = Pixel_Lacunarity(model_name=model_name, scales=scales, kernel=(kernel, kernel), stride =(stride, stride), bias=bias)
            elif poolingLayer == "ScalePyramid_Lacunarity":
                m.global_pool = ScalePyramid_Lacunarity(model_name=model_name, num_levels=num_levels, sigma = sigma, min_size = min_size, kernel=(kernel, kernel), stride =(stride, stride))
            elif poolingLayer == "BuildPyramid":
                m.global_pool = BuildPyramid(model_name=model_name, num_levels=num_levels, kernel=(kernel, kernel), stride =(stride, stride))
            elif poolingLayer == "DBC":
                m.global_pool = DBC(model_name=model_name, r_values = scales, window_size = kernel)
            elif poolingLayer == "GDCB":
                m.global_pool = GDCB(3,5)
        
        elif aggFunc == "global":
            if poolingLayer == "max":
                m.global_pool = nn.AdaptiveMaxPool2d((1,1))
            elif poolingLayer == "avg":                                                                                                                                                                                                                            
                m.global_pool = nn.Adaptiveglobal_pool2d((1, 1))
            elif poolingLayer == "Base_Lacunarity":
                m.global_pool = Base_Lacunarity(model_name=model_name, scales=scales,bias=bias)
            elif poolingLayer == "Pixel_Lacunarity":
                m.global_pool = Pixel_Lacunarity(model_name=model_name, scales=scales, bias=bias)
            elif poolingLayer == "ScalePyramid_Lacunarity":
                m.global_pool = ScalePyramid_Lacunarity(model_name=model_name, num_levels=num_levels, sigma = sigma, min_size = min_size)
            elif poolingLayer == "BuildPyramid":
                m.global_pool = BuildPyramid(model_name=model_name, num_levels=num_levels)
            elif poolingLayer == "DBC":
                m.global_pool = DBC(model_name=model_name, r_values = scales, window_size = kernel)
            elif poolingLayer == "GDCB":
                m.global_pool = GDCB(3,5)