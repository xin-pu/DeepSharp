using DeepSharp.RL.Models;
using Action = DeepSharp.RL.Models.Action;

namespace TorchSharpTest.RLTest
{
    /// <summary>
    ///     多臂赌博机,每个赌博机以 0,0.1到1 的概率随机生成
    /// </summary>
    public class KArmedBandit : Envir
    {
        public KArmedBandit(int k)
        {
            K = k;
            var random = new Random();
            bandits = new Bandit[k];
            foreach (var i in Enumerable.Range(0, k))
                bandits[i] = new Bandit($"{i}", random.Next(2, 8) * 1f / 10);
            State = new State {Value = torch.zeros(k)};
            Reward = new Reward(torch.zeros(k));
        }

        protected int K { set; get; }
        protected Bandit[] bandits { set; get; }

        /// <summary>
        /// </summary>
        /// <param name="action">动作，该环境下包含智能体选择的赌博机索引</param>
        /// <returns>返回选择的赌博机当次执行后获得的金币数量 0 或 1</returns>
        public override State UpdateState(Action action)
        {
            var banditSelectIndex = action.Value.item<long>();
            var bandit = bandits[banditSelectIndex];
            var value = bandit.Step();
            var stateArray = Enumerable.Repeat(0, K).ToArray();
            stateArray[banditSelectIndex] = value;
            var stateTensor = torch.from_array(stateArray, torch.ScalarType.Float32);
            return new State {Value = stateTensor};
        }

        /// <summary>
        ///     该环境下 当次奖励为赌博机的获得金币数量，无需转换
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public override Reward GetReward(State state)
        {
            var sum = state.Value.sum();
            return new Reward(sum);
        }


        public override string ToString()
        {
            var final = string.Join("\r\n", bandits.Select(a => a.ToString()));
            return final;
        }
    }
}