import hydra

from src import Registration

@hydra.main(version_base=None, config_name='config', config_path='config')
def main(config):
    optimizer = Registration(config.model)
    optimizer.fit()

if __name__ == "__main__":
    main()