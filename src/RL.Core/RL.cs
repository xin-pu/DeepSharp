namespace RL.Core
{
    /// <summary>
    ///     生成每个Action的概率，并根据概率 选择当次Action
    /// </summary>
    public class ActionPickUp
    {
        /// <summary>
        ///     没有放回，按权重采样
        /// </summary>
        /// <param name="probs"></param>
        /// <param name="sampleOut"></param>
        /// <returns></returns>
        public static torch.Tensor PickUp(torch.Tensor probs, int sampleOut = 1)
        {
            var r = torch.multinomial(probs, sampleOut);
            return r;
        }

        /// <summary>
        ///     有放回，按权重采样
        /// </summary>
        /// <param name="probs"></param>
        /// <param name="sampleOut"></param>
        /// <returns></returns>
        public static torch.Tensor PickUpWithReplace(torch.Tensor probs, int sampleOut = 1)
        {
            var r = torch.multinomial(probs, sampleOut, true);
            return r;
        }
    }
}