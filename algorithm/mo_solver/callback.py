import os
import numpy as np
import matplotlib.pyplot as plt
from utils import normalize_y, denormalize_y

class RecordCallback:
    def __init__(self, real_problem, surrogate_problem, args, iters_to_record=100):
        self.args = args
        self.iters_to_record = iters_to_record
        self.X = []
        self.Y = []
        self.Y_real = []
        self.surrogate_problem = surrogate_problem
        self.real_problem = real_problem
        # try:
        #     self.pareto_front = real_problem.get_pareto_front()
        #     if self.args.normalize_y:
        #         self.pareto_front = normalize_y(self.args, self.pareto_front)
        # except:
        #     pass
        self.save_dir = os.path.join(self.args.results_dir, 'runtime_record')

    def _do(self, algorithm):
        try:
            if algorithm.n_iter % self.iters_to_record != 0:
                return 
        except:
            if algorithm.n_gen % self.iters_to_record != 0:
                return
        
        x = algorithm.pop.get('X')
        y = algorithm.pop.get('F')

        if self.args.normalize_y:
            y = denormalize_y(self.args, y)

        if self.args.discrete:
            from utils import to_integers
            # res_x = res_x.reshape((-1,) + input_shape)
            input_shape = self.args.input_shape
            x = x.reshape((-1,) + input_shape)
            x = to_integers(x)

        self.X.append(x)
        self.Y.append(y)
        
        y_real = self.real_problem.evaluate(x)
        self.Y_real.append(y_real)

        try:
            save_dir = os.path.join(self.save_dir, f'{algorithm.n_iter}')
        except:
            save_dir = os.path.join(self.save_dir, f'{algorithm.n_gen}')
        os.makedirs(save_dir, exist_ok=True)
        np.save(arr=x, file=os.path.join(save_dir, 'x.npy'))
        np.save(arr=y, file=os.path.join(save_dir, 'y_surrogate.npy'))
        np.save(arr=y_real, file=os.path.join(save_dir, 'y_predict.npy'))

        # self._plot(y, y_real, save_dir)
        self.plot_all()
        # self

    def _plot(self, y, y_real, save_dir):
        n_obj = len(y[0])
        y2figname = [(y, 'y_surrogate'),
                     (y_real, 'y_predict')]
        
        for y0, figname in y2figname:
            if n_obj == 2:
                plt.figure(figsize=(8, 8))
                plt.scatter(y0[:, 0], y0[:, 1], color='blue', s=10)
            elif n_obj == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(y0[:, 0], y0[:, 1], y0[:, 2], color='blue')
                ax.set_ylim([np.max(y0[:, 1]), np.min(y0[: 1])])

            else:
                fig, axs = plt.subplots(n_obj, n_obj, figsize=(20, 20))
                for i in range(n_obj):
                    for j in range(n_obj):
                        if i == j:
                            continue
                        ax = axs[i, j]
                        ax.set_title(f'obj.{i + 1} and obj.{j + 1}')
                        ax.set_xlabel(f'f{i}')
                        ax.set_ylabel(f'f{j}')
                        ax.scatter(y0[:, i], y0[:, j], color='blue')
                plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
            plt.savefig(os.path.join(save_dir, figname + '.png'))
            plt.savefig(os.path.join(save_dir, figname + '.pdf'))


    def plot_all(self):
        color_set = {
            'Amaranth': np.array([0.9, 0.17, 0.31]), 
            'Amber': np.array([1.0,0.49,0.0]),  
            'Bleu de France': np.array([0.19,0.55,0.91]),
            'Electric violet': np.array([0.56, 0.0, 1.0]),
            'Arsenic': np.array([0.23, 0.27, 0.29]),
            'Blush': np.array([0.87, 0.36, 0.51]),
            'Dark sea green': np.array([0.56,0.74,0.56]),
            'Dark electric blue': np.array([0.33,0.41,0.47]),
            'Dark gray': np.array([0.66, 0.66, 0.66]),
            'French beige': np.array([0.65, 0.48, 0.36]),
            'Grullo': np.array([0.66, 0.6, 0.53]),
            'Dark coral': np.array([0.8, 0.36, 0.27]),
            'Old lavender': np.array([0.47, 0.41, 0.47]),
            'Sandy brown': np.array([0.96, 0.64, 0.38]),
            'Dark cyan': np.array([0.0, 0.55, 0.55]),
            'Brick red': np.array([0.8, 0.25, 0.33]),
            'Dark pastel green': np.array([0.01, 0.75, 0.24])
        }

        n_obj = len(self.Y[0][0])
        # print(self.Y)
        # print(self.Y[0])
        # print(self.Y[0][0])
        # n_obj = self.Y
        y2axs = [(self.Y, 'y_surrogate', 0, 0), (self.Y_real, 'y_predict', 1, 0)]
        if n_obj == 2:
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            for y, name, i, j in y2axs:
                ax = axs[i, j]
                for idx, y_i in enumerate(y):
                    
                    color = color_set[list(color_set.keys())[idx % len(color_set.keys())]]
                    ax.set_title(f'{name} for {self.args.env_name}')
                    ax.scatter(y_i[:, 0], y_i[:, 1], color=color, s=10, label=name)
                    ax.set_xlabel('f1')
                    ax.set_ylabel('f2')
            
            ax0 = axs[0, 1]
            x = np.array([p for p in range(10)])
            for idx in range(len(self.Y)):
                color = color_set[list(color_set.keys())[idx % len(color_set.keys())]]
                ax0.plot(x, x+idx, color=color, markersize=5, label=f'{idx * self.iters_to_record}')
            ax0.legend()
            
            plt.savefig(os.path.join(self.save_dir, 'plot_all.png'))
            plt.savefig(os.path.join(self.save_dir, 'plot_all.pdf'))

        else:
            return

        