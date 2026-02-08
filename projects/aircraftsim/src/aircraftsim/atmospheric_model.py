class AtmosphericModel:
    pass


class AtmosphericModelBuilder:
    def build_atmospheric_model(self, config_file) -> AtmosphericModel:
        raise NotImplementedError
