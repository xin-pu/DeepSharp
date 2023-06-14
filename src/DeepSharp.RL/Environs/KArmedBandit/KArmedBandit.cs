using System.Text;
using DeepSharp.RL.Environs.Spaces;
using MathNet.Numerics.Random;

namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     多臂赌博机,每个赌博机以 0,0.1到1 的概率随机生成
    /// </summary>
    public sealed class KArmedBandit : Environ<Space, Space>
    {
        public KArmedBandit(int k, DeviceType deviceType = DeviceType.CUDA)
            : base("KArmedBandit", deviceType)
        {
            bandits = new Bandit[k];
            ActionSpace = new Disperse(k, deviceType: deviceType);
            ObservationSpace = new Box(0, 1, new long[] {k}, deviceType);
            Create(k);
            Reset();
        }

        private Bandit[] bandits { get; }
        public Bandit this[int k] => bandits[k];


        private void Create(int k)
        {
            var random = new SystemRandomSource();
            foreach (var i in Enumerable.Range(0, k))
                bandits[i] = new Bandit($"{i}", random.NextDouble());
        }


        /// <summary>
        ///     该环境下 当次奖励为赌博机的获得金币数量，无需转换
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public override Reward GetReward(Observation observation)
        {
            var sum = observation.Value!.to_type(torch.ScalarType.Float32)
                .sum()
                .item<float>();
            var reward = new Reward(sum);
            return reward;
        }


        /// <summary>
        /// </summary>
        /// <param name="act">动作，该环境下包含智能体选择的赌博机索引</param>
        /// <returns>返回选择的赌博机当次执行后获得的金币数量 0 或 1</returns>
        public override Observation Update(Act act)
        {
            var obs = new float[ObservationSpace!.N];
            var index = act.Value!.ToInt64();
            var bandit = bandits[index];
            var value = bandit.Step();
            obs[index] = value;

            var obsTensor = torch.from_array(obs, torch.ScalarType.Float32).to(Device);
            return new Observation(obsTensor);
        }


        /// <summary>
        ///     Discount Reward
        ///     该环境无奖励折扣
        /// </summary>
        /// <param name="episode"></param>
        /// <param name="gamma"></param>
        /// <returns></returns>
        public override float DiscountReward(Episode episode, float gamma)
        {
            return 1;
        }

        /// <summary>
        ///     没满20次采样，环境关闭
        /// </summary>
        /// <param name="epoch"></param>
        /// <returns></returns>
        public override bool IsComplete(int epoch)
        {
            return epoch >= 20;
        }


        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine(base.ToString());
            str.Append(string.Join("\r\n", bandits.Select(a => $"\t{a}")));
            return str.ToString();
        }
    }
}