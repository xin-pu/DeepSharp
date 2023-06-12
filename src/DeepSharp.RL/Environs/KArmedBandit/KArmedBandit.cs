using DeepSharp.RL.Environs.Spaces;
using DeepSharp.RL.Models;

namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     多臂赌博机,每个赌博机以 0,0.1到1 的概率随机生成
    /// </summary>
    public sealed class KArmedBandit : Environ<Space, Space>
    {
        public KArmedBandit(int k, torch.Device device)
            : base("KArmedBandit", device)
        {
            ObservationSpace = new Box(0, 1, new long[] {k});
            ActionSpace = new Box(0, 1, new long[] {k});
            var random = new Random();
            bandits = new Bandit[k];
            foreach (var i in Enumerable.Range(0, k))
                bandits[i] = new Bandit($"{i}", random.Next(2, 8) * 1f / 10);

            Observation = new Observation(torch.zeros(k));
            Reward = new Reward(0);
            Reset();
        }

        public KArmedBandit(int k, DeviceType device)
            : this(k, new torch.Device(device))
        {
        }

        public Bandit this[int k] => bandits[k];

        private Bandit[] bandits { get; }

        /// <summary>
        ///     该环境下 当次奖励为赌博机的获得金币数量，无需转换
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override Reward GetReward(Observation observation)
        {
            var sum = observation.Value!.sum().item<float>();
            return new Reward(sum);
        }

        /// <summary>
        /// </summary>
        /// <param name="act">动作，该环境下包含智能体选择的赌博机索引</param>
        /// <returns>返回选择的赌博机当次执行后获得的金币数量 0 或 1</returns>
        public override Observation UpdateEnviron(Act act)
        {
            var obs = new float[ObservationSpace.N];

            var banditSelectIndex = act.Value!.data<long>().ToArray();

            foreach (var index in banditSelectIndex)
            {
                var bandit = bandits[index];
                var value = bandit.Step();
                obs[index] = value;
            }

            var obsTensor = torch.from_array(obs, torch.ScalarType.Float32).to(Device);
            return new Observation(obsTensor);
        }

        public override Act Sample()
        {
            throw new NotImplementedException();
        }

        public override float DiscountReward(Episode episode, float gamma)
        {
            return 1;
        }

        public override bool StopEpoch(int epoch)
        {
            return epoch % 20 == 0;
        }


        public override string ToString()
        {
            var final = string.Join("\r\n", bandits.Select(a => a.ToString()));
            return final;
        }
    }
}