namespace TorchSharpTest.RLTest
{
    /// <summary>
    ///     简化的赌博机，以Prob 概率吐出一枚硬币
    /// </summary>
    public class Bandit
    {
        public double Prob { internal set; get; }

        public Bandit(string name, double prob = 0.7)
        {
            Name = name;
            Prob = prob;
        }

        public string Name { get; set; }


        public int Step()
        {
            var d = new Random();
            var pro = d.NextDouble();
            return pro <= Prob ? 1 : 0;
        }

        public override string ToString()
        {
            return $"Bandit{Name}:\t{Prob:P}";
        }
    }
}