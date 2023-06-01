using DeepSharp.RL.Models;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     多臂赌博机,每个赌博机以 0,0.1到1 的概率随机生成
    /// </summary>
    public class KArmedBandit : Environ
    {
        public KArmedBandit(int k) : base("KArmedBandit")
        {
            ObservationSpace = k;
            ActionSpace = k;
            var random = new Random();
            bandits = new Bandit[k];
            foreach (var i in Enumerable.Range(0, k))
                bandits[i] = new Bandit($"{i}", random.Next(2, 8) * 1f / 10);

            bandits[0].Prob = 0.3;
            bandits[1].Prob = 0.9;
            Observation = new Observation(torch.zeros(k));
            Reward = new Reward(0);
            Reset();
        }


        protected Bandit[] bandits { set; get; }


        /// <summary>
        /// </summary>
        /// <param name="action">动作，该环境下包含智能体选择的赌博机索引</param>
        /// <returns>返回选择的赌博机当次执行后获得的金币数量 0 或 1</returns>
        public override Observation UpdateEnviron(Action action)
        {
            var obs = new float[ObservationSpace];

            var banditSelectIndex = action.Value.data<long>().ToArray();

            foreach (var index in banditSelectIndex)
            {
                var bandit = bandits[index];
                var value = bandit.Step();
                obs[index] = value;
            }

            var obsTensor = torch.from_array(obs, torch.ScalarType.Float32);
            return new Observation(obsTensor);
        }

        /// <summary>
        ///     该环境下 当次奖励为赌博机的获得金币数量，无需转换
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override Reward GetReward(Observation observation)
        {
            var sum = observation.Value.sum().item<float>();
            return new Reward(sum);
        }


        public override string ToString()
        {
            var final = string.Join("\r\n", bandits.Select(a => a.ToString()));
            return final;
        }
    }
}