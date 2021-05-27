from train import *
import matplotlib.pyplot as plt
import seaborn as sns
import time
from synthdata import SynthDataLoader
from baseline_completions import *
from cond_plurality import Conditional_plurality

def neg_1_model(x):
    return x + (torch.abs(x)) - 1

if __name__ == '__main__':
    sns.set_theme()
    parser = argparse.ArgumentParser(description='Ballot completion using an AE.')
    parser.add_argument('--train_split', '-sp', type=float, default=0.8,
                    help='What percentage of the data should be used as training data.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='number of epochs to train for.')
    parser.add_argument('--repeats', '-r', type=int, default=5,
                        help='How often to repeat each experiment.')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', '-m', type=str, default='ae')
    args = parser.parse_args()
    arg_model = args.model

    start = time.time()
    # dataset = RealDataLoader('poland_warszawa_2019_ursynow.pb', dropout = 0)
    dataset = SynthDataLoader(num_categories=5, num_voters=8000, num_projects=80, dropout=0, prior=0.6)
    verbose=False

    dropouts = np.arange(0, 1, 0.05)
    model_accs = []
    model_outcomes = []
    baseline_accs = []
    baseline_outcomes = []
    for dropout in dropouts:

        print('DROPOUT:', dropout)

        if dropout == 0:
            model_accs.append(1)
            model_outcomes.append(1)
            baseline_accs.append(1)
            baseline_outcomes.append(1)
            continue

        m_accs = []
        m_outcomes = []
        b_accs = []
        b_outcomes = []
        for i in range(args.repeats):
            print(f'repeat: {i}')
            # Create new x tensors
            # Create dropout per project
            drop_pp = torch.randn(dataset.targets.shape[1]) * 0.5
            # Scale between 0 and 1 with mean dropout
            if torch.min(drop_pp) < -dropout:
                drop_pp *= -dropout / torch.min(drop_pp)
            if torch.max(drop_pp) > 1 - dropout:
                drop_pp *= (1-dropout) / torch.max(drop_pp)
            drop_pp += dropout

            # print(torch.min(drop_pp), torch.mean(drop_pp), torch.max(drop_pp))
            dataset.dropout = drop_pp



            dataset.create_x_from_dropout()
            num_ballots = len(dataset)
            num_val = int(num_ballots*(1-args.train_split)) + 1 
            num_train = num_ballots - num_val
            train_dataset, val_dataset = data.random_split(dataset, [num_train ,num_val])
            if arg_model == 'ae':
                model = AutoEncoder(dataset.n_projects, [54], 50)
                model.train()
                train(model, train_dataset, epochs=args.epochs, batch_size=256, verbose = verbose, pc = None, device=args.device, use_mask=True)
                model.eval()
            elif arg_model == 'cp':
                n_projects = dataset.n_projects
                project_costs = dataset.project_costs
                model = Conditional_plurality(train_dataset, n_projects, project_costs, use_costs=False)
            else:
                print('Enter valid model')
                exit()

            m_accs.append(evaluate_acc(model, val_dataset, verbose=verbose))
            m_outcomes.append(evaluate_outcome(model, dataset, val_dataset, verbose=verbose))

            # partial_model = get_partial_model(train_dataset=train_dataset, nn_fraction=0.1)
            b_accs.append(evaluate_acc(neg_1_model, val_dataset, verbose=verbose))
            b_outcomes.append(evaluate_outcome(neg_1_model, dataset, val_dataset, is_function = True, verbose=verbose))

        model_accs.append(np.mean(m_accs))
        model_outcomes.append(np.mean(m_outcomes))
        baseline_accs.append(np.mean(b_accs))
        baseline_outcomes.append(np.mean(b_outcomes))

    end = time.time()
    duration = end-start
    print(f'Took {duration//60} minutes and {duration%60} seconds')
    fig, (ax1, ax2) = plt.subplots(2)
    plt.tight_layout()

    if arg_model=='ae':
        legend_name = 'AutoEncoder'
    elif arg_model=='cp':
        legend_name = 'Conditional plurality'


    ax1.plot(dropouts, model_accs)
    ax1.plot(dropouts, baseline_accs)
    ax1.set_xlabel('dropout')
    ax1.set_ylabel('accuracy')
    ax1.legend([legend_name, 'Baseline'])

    ax2.plot(dropouts, model_outcomes)
    ax2.plot(dropouts, baseline_outcomes)
    ax2.set_xlabel('dropout')
    ax2.set_ylabel('% of budget allocated the same')
    ax2.legend([legend_name, 'Baseline'])

    plt.show()
