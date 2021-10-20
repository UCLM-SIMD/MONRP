import random
from typing import List
from algorithms.EDA.eda_algorithm import EDAAlgorithm
from evaluation.format_population import format_population
from algorithms.abstract_default.evaluation_exception import EvaluationLimit
from evaluation.update_nds import get_nondominated_solutions
from models.solution import Solution
from models.problem import Problem
from datasets.dataset_gen_generator import generate_dataset_genes
from algorithms.GRASP.GraspSolution import GraspSolution
from algorithms.GRASP.Dataset import Dataset
from algorithms.EDA.UMDA.umda_executer import UMDAExecuter
from algorithms.genetic.abstract_genetic.basegenetic_algorithm import BaseGeneticAlgorithm
from algorithms.genetic.genetic.genetic_executer import GeneticExecuter
from algorithms.genetic.genetic.genetic_utils import GeneticUtils

import copy
import time
import numpy as np
import math
from scipy import stats as scipy_stats


class MIMICAlgorithm(EDAAlgorithm):
    def __init__(self, dataset_name:str="test", random_seed:int=None, debug_mode:bool=False, tackle_dependencies:bool=False,
                population_length:int=100, max_generations:int=100, max_evaluations:int=0,
                selected_individuals:int=60, selection_scheme:str="nds", replacement_scheme:str="replacement"):

        # self.executer = UMDAExecuter(algorithm=self)
        super().__init__(dataset_name, random_seed, debug_mode, tackle_dependencies,
            population_length, max_generations, max_evaluations)

        #self.dataset = Dataset(dataset_name)
        #self.dataset_name = dataset_name
        self.gene_size: int = len(self.dataset.pbis_cost)

        #self.population_length: int = population_length
        #self.max_generations: int = max_generations
        #self.max_evaluations: int = max_evaluations

        self.selected_individuals: int = selected_individuals

        self.selection_scheme:str = selection_scheme
        self.replacement_scheme:str = replacement_scheme

        #self.nds = []
        #self.num_evaluations: int = 0
        #self.num_generations: int = 0
        #self.best_individual = None

        self.population:List[GraspSolution] = []

        #self.debug_mode = debug_mode
        #self.tackle_dependencies = tackle_dependencies

        #self.random_seed = random_seed
        #if random_seed is not None:
        #    np.random.seed(random_seed)

        self.file:str = str(self.__class__.__name__)+"-"+str(dataset_name)+"-"+str(random_seed)+"-"+str(population_length)+"-" +\
            str(max_generations) + "-"+str(max_evaluations)+".txt"

    def get_name(self):
        return f"MIMIC selection{self.selection_scheme} {self.replacement_scheme}"

    '''
    LEARN PROBABILITY MODEL
    '''

    def learn_probability_model(self, population, selected_individuals):
        # init structures
        parents = np.zeros(self.gene_size, dtype=int)
        used = np.full(self.gene_size, False)
        variables = np.zeros(self.gene_size, dtype=int)
        conditionals = np.zeros((self.gene_size, 2), dtype=float)

        marginals = self.learn_marginals(population, selected_individuals)

        # Obtain entropies.
        entropies = np.zeros(self.gene_size)
        for i in range(self.gene_size):
            entropies[i] = self.get_entropy(
                population, i, selected_individuals)

        # Takes the variable with less entropy as the first.
        current_var = np.argmin(entropies)
        parents[0] = -1
        variables[0] = current_var

        # Marks it as used.
        used[current_var] = True

        # Adds iteratively the variable with less conditional entropy.
        for i in range(1,self.gene_size):
            # Chooses the next variable.
            parents[i] = current_var
            current_var = self.get_lower_conditional_entropy(
                population, current_var, used, selected_individuals)
            variables[i] = current_var
            used[current_var] = True
            prob_x, prob_y, prob_xy = self.get_distributions(
                population, current_var, parents[i], selected_individuals)
            conditionals[i][0] = prob_xy[1][0]
            conditionals[i][1] = prob_xy[1][1]

        return marginals, parents, variables, conditionals

    def learn_marginals(self, population, selected_individuals, laplace=0):
        '''
            // Learns a model from the best individuals in a given population
            // It assumes that population is sorted in descendent order.
            public void learnMarginals(Population pPopulation, int nIndividuals) {
                int sizePop = pPopulation.getPopSize();
                int selected = nIndividuals;
                int[][] data = pPopulation.getElements();
                // Learns the model.
                Arrays.fill(marginals,0);
                for (int i=0;i<selected;i++)
                    for (int j=0;j<size;j++)
                        if (data[i][j]==1) marginals[j]+=1;
                // Uses Laplace normalization.
                for (int j=0;j<size;j++)
                    marginals[j]=(marginals[j]+laplace)/(selected+(2*laplace));
            }
        '''
        marginals = np.zeros(self.gene_size)
        # if fixed number -> self.selected_individuals. if selection by NDS ->unknown ->len
        #selected_individuals = len(population)
        for i in range(selected_individuals):
            for j in range(self.gene_size):
                if population[i].selected[j] == 1:
                    marginals[j] += 1
        for j in range(self.gene_size):
            marginals[j] = (marginals[j]+laplace) / \
                (selected_individuals+(2*laplace))
        return marginals

    def get_probability_distribution(self, elements, v1, N, laplace=1):
        prob = np.zeros(2)
        #N = len(elements)
        for i in range(N):
            prob[elements[i].selected[v1]] += 1.0
        for i in range(2):
            if laplace == 1:
                prob[i] = (prob[i]+1)/N+2
            else:
                prob[i] = (prob[i])/N
        return prob

    def get_entropy(self, elements, var1, N):
        '''
           public double getEntropy(int var1,int N){
            int i;
            double entropy=0.0;
            double[] prob;
            prob = getProbabilityDistributionOf(var1,N,0);
            for(i=0;i<2;i++)
                if (prob[i] > 0.0) entropy +=  prob[i] * Utils.log2(prob[i]);
            if (entropy != 0.0) entropy *= -1.0;
            return entropy;
        }
        '''
        # entropy = 0
        probs = self.get_probability_distribution(elements, var1, N, 0)
        # for i in range(2):
        #    if probs[i] > 0.0:
        #        entropy += probs[i]*math.log2(probs[i])
        #    if entropy != 0.0:
        #        entropy *= -1.0
        # return entropy
        return scipy_stats.entropy(probs, base=2)

    def get_conditional_entropy(self, population, var1, var2, N):
        '''
           public double getConditionalEntropy(int var1,int var2,int N){
            int i,j,valueI,valueJ;
            double pValue,entropy=0.0,entropy2;
            double[] probParent;
            double[][] probXY;
            ThreeProbs tp;
            tp = getDistributionsOf(var1,var2,N,0);
            probParent = tp.getPY();
            probXY = tp.getPXY();
            for(j=0;j<2;j++){
                entropy2=0.0;
                for(i=0;i<2;i++){
                    if (probXY[i][j]>0.0) entropy2 += probXY[i][j] * Utils.log2(probXY[i][j]);
                }
                if (entropy2 != 0.0 ) entropy2 *= -1.0;
                entropy += probParent[j] * entropy2;
            }
            return entropy;
        }
        '''
        entropy = 0
        prob_x, prob_y, prob_xy = self.get_distributions(
            population, var1, var2, N, 1)
        for j in range(2):
            entropy2 = 0.0
            for i in range(2):
                if(prob_xy[i][j] > 0):
                    entropy2 += prob_xy[i][j]*math.log2(prob_xy[i][j])
            if entropy2 != 0:
                entropy2 *= -1
            entropy += prob_y[j]*entropy2
        return entropy

    def get_lower_conditional_entropy(self, population, parent, used, N):
        '''
        private int getLowerConditionalEntropy(int parent, boolean[] used){
            int index = -1;
            double CE;
            double minCE = Double.POSITIVE_INFINITY;
            for (int i=0;i<size;i++){
                if (used[i]) continue;
                CE = data.getConditionalEntropy(parent,i,N);
                if (CE<minCE){
                    minCE = CE;
                    index = i;
                }
            }
            return index;
        }
        '''
        index = -1
        min_ce = float("inf")
        for i in range(self.gene_size):
            if(used[i]):
                continue
            ce = self.get_conditional_entropy(population, parent, i, N)
            if(ce < min_ce):
                min_ce = ce
                index = i
        return index

    def get_distributions(self, population, X, Y, N, laplace=1):
        '''
        public ThreeProbs getDistributionsOf(int X,int Y, int N, int Laplace){
            double[][] probXY = new double[2][2];
            double[] probX = new double[2];
            double[] probY = new double[2];
            ThreeProbs tp = new ThreeProbs(X,Y);
            int i,j,row;
            int[] numY = new int[2];
            for(j=0;j<2;j++){
                numY[j]=0;
                probY[j]=0.0;
            }
            for(i=0;i<2;i++){
                probX[i]=0.0;
                for(j=0;j<2;j++)
                    probXY[i][j]=0.0;
            }
            for(row=0;row<N;row++){
                probXY[elements[row][X]][elements[row][Y]] +=1.0;
                numY[elements[row][Y]]++;
                probX[elements[row][X]] += 1.0;
                probY[elements[row][Y]] += 1.0;
            }
            for(i=0;i<2;i++){
                if (Laplace==1) probX[i] = (probX[i]+1.0)/(double)(N+2);
                else probX[i] = probX[i]/(double)(N);
                for(j=0;j<2;j++)
                    if (Laplace==1)
                        probXY[i][j] = (probXY[i][j]+1.0)/(double)(numY[j]+2);
                    else probXY[i][j] = (probXY[i][j])/(double)(numY[j]);
            }
            for(j=0;j<2;j++)
                if (Laplace==1) probY[j] = (probY[j]+1.0)/(double)(N+2);
                else probY[j] = (probY[j])/(double)(N);
            tp.setPX(probX);
            tp.setPY(probY);
            tp.setPXY(probXY);

            return tp;
        }
        '''
        num_y = np.zeros(2)
        prob_y = np.zeros(2)
        prob_x = np.zeros(2)
        prob_xy = np.zeros((2, 2))
        for row in range(N):
            prob_x[population[row].selected[X]] += 1
            prob_y[population[row].selected[Y]] += 1
            prob_xy[population[row].selected[X]
                    ][population[row].selected[Y]] += 1
            num_y[population[row].selected[Y]] += 1

        for i in range(2):
            if laplace == 1:
                prob_x[i] = (prob_x[i]+1.0)/(N+2)
            else:
                prob_x[i] = prob_x[i]/N
            for j in range(2):
                if laplace == 1:
                    prob_xy[i][j] = (prob_xy[i][j]+1.0)/(num_y[j]+2)
                else:
                    prob_xy[i][j] = prob_xy[i][j]/num_y[j]

        for i in range(2):
            if laplace == 1:
                prob_y[i] = (prob_y[i]+1.0)/(N+2)
            else:
                prob_y[i] = prob_y[i]/N

        return prob_x, prob_y, prob_xy

    '''
    SAMPLE NEW POPULATION
    '''

    def sample_new_population(self, marginals, parents, variables, conditionals):
        new_population = []
        for i in np.arange(self.population_length):
            new_individual = self.generate_sample(
                marginals, parents, variables, conditionals)
            new_population.append(new_individual)
        return new_population

    def generate_sample(self, marginals, parents, variables, conditionals):
        '''
        // Samples an individual
        public int[] getSample(){
            int sample[] = new int[size];
            for (int j = 0; j < size; j++) {
                if (parents[j] == -1) {
                    if (generator.nextDouble() < marginals[variables[j]])
                        sample[variables[j]] = 1;
                    else
                        sample[variables[j]] = 0;
                    }
                else {
                    // a partir del segundo elemento que ya tiene padre:

                    // coge el condicional de la fila del gen que toque (j)
                    // busca
                    // parents[j] = padre del elemento actual = 4 p.e.
                    // sample[parents[j]] = 1 o 0 -> sample[4]=0 (ya lo tenemos calculado en alguna iteracion anterior)
                    // conditionals[j][sample[parents[j]]] = conditionals[j][0]
                    // obtenemos para cada gen j el condicional segun si el sample del padre era 1 o 0
                    if (generator.nextDouble() < conditionals[j][sample[parents[j]]])
                    // sample[variables[j]] -> variables[j], si j=1 ->variables[1]=4
                    // sample[4]
                        sample[variables[j]] = 1;
                    else
                        sample[variables[j]] = 0;
                    }
                }
            return sample;
        }
        '''
        sample = np.zeros(self.gene_size, dtype=int)
        for j in range(self.gene_size):
            if(parents[j] == -1):
                if(random.random() < marginals[variables[j]]):
                    sample[variables[j]] = 1
                else:
                    sample[variables[j]] = 0

            else:
                if(random.random() < conditionals[j][sample[parents[j]]]):
                    sample[variables[j]] = 1
                else:
                    sample[variables[j]] = 0

        sample_ind = GraspSolution(self.dataset,None, selected=sample)
        return sample_ind

    # RUN ALGORITHM------------------------------------------------------------------
    def run(self):
        self.reset()
        paretos = []
        start = time.time()

        returned_population = None
        self.population = self.generate_initial_population()
        self.evaluate(self.population, self.best_individual)
        try:
            while (not self.stop_criterion(self.num_generations, self.num_evaluations)):
                # selection
                individuals = self.select_individuals(self.population)
                # learning
                # guardar el num individuos seleccionados porque puede variar
                marginals, parents, variables, conditionals = self.learn_probability_model(
                    individuals, len(individuals))

                # replacement
                self.population = self.sample_new_population(
                    marginals, parents, variables, conditionals)

                # repair population if dependencies tackled:
                if(self.tackle_dependencies):
                    self.population = self.repair_population_dependencies(
                        self.population)

                # evaluation
                self.evaluate(self.population, self.best_individual)

                # update nds with solutions constructed and evolved in this iteration
                get_nondominated_solutions(self.population, self.nds)

                self.num_generations += 1

                if self.debug_mode:
                    paretos.append(format_population(self.nds, self.dataset))

        except EvaluationLimit:
            pass

        #self.nds = format_population(self.nds, self.dataset)
        end = time.time()

        print("\nNDS created has", self.nds.__len__(), "solution(s)")

        return {"population": self.nds,
                "time": end - start,
                "numGenerations": self.num_generations,
                "best_individual": self.best_individual,
                "numEvaluations": self.num_evaluations,
                "paretos": paretos
                }


#if __name__ == '__main__':
#    algorithm = MIMICAlgorithm(dataset_name="1", population_length=4, max_generations=3,
#                               max_evaluations=0, selected_individuals=4, selection_scheme="nds", replacement_scheme="replacement")
#    result = algorithm.run()
