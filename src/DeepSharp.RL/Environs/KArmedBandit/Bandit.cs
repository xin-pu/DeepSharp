using MathNet.Numerics.Random;

namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     简化的赌博机，以Prob 概率吐出一枚硬币
    /// </summary>
    public class Bandit
    {
        public Bandit(string name, double prob = 0.7)
        {
            Name = name;
            Prob = prob;
            RandomSource = new SystemRandomSource();
        }

        protected SystemRandomSource RandomSource { set; get; }

        public double Prob { set; get; }

        public string Name { get; set; }


        public float Step()
        {
            var pro = RandomSource.NextDouble();
            return pro <= Prob ? 1 : 0;
        }

        public override string ToString()
        {
            return $"Bandit{Name}:\t{Prob:P}";
        }
    }
}