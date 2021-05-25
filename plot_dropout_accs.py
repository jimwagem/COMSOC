from train import *
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    sns.set_theme()
    parser = argparse.ArgumentParser(description='Ballot completion using an AE.')
    parser.add_argument('--train_split', '-sp', type=float, default=0.8,
                    help='What percentage of the data should be used as training data.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs to train for.')
    args = parser.parse_args()

    dropouts = np.arange(0, 1, 0.05)
    accs = []
    outcomes = []
    for dropout in dropouts:
        if dropout == 0:
            accs.append(1)
            outcomes.append(1)
            continue
        print('DROPOUT:', dropout)
        dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout = dropout)
        num_ballots = len(dataset)
        num_val = int(num_ballots*args.train_split)
        num_train = num_ballots - num_val
        train_dataset, val_dataset = data.random_split(dataset, [num_train ,num_val])

        model = AutoEncoder(dataset.n_projects, [75], 50)
        train(model, train_dataset, epochs=args.epochs, batch_size=8, verbose = False, pc = dataset.project_costs)
        accs.append(evaluate_acc(model, val_dataset))
        outcomes.append(evaluate_outcome(model, dataset, val_dataset))

    plt.plot(dropouts, accs)
    plt.plot(dropouts, outcomes)
    plt.xlabel('dropout')
    plt.ylabel('accuracy')
    plt.legend(['accuracy', 'budget correctly allocated'])
    plt.show()
