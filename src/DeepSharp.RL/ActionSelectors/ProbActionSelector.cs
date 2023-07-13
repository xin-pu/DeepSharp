namespace DeepSharp.RL.ActionSelectors
{
    /// <summary>
    ///     输入归一化概率,返回从分布中采样的结果
    /// </summary>
    public class ProbActionSelector : ActionSelector
    {
        /// <summary>
        /// </summary>
        /// <param name="probs">such as [[0.8,0.1,0.1],[0.01,0.98,0.01]]</param>
        /// <returns></returns>
        public override torch.Tensor Select(torch.Tensor probs)
        {
            var dims = probs.dim();

            return dims switch
            {
                1 => GetActionByDim1(probs),
                2 => GetActionByDim2(probs),
                _ => throw new NotSupportedException("Support Dim which 1 & 2")
            };
        }

        private torch.Tensor GetActionByDim1(torch.Tensor probs)
        {
            return torch.multinomial(probs, 1);
        }

        private torch.Tensor GetActionByDim2(torch.Tensor probs)
        {
            var width = (int) probs.shape[0];

            var arr = Enumerable.Range(0, width).Select(i =>
            {
                var tensorIndices = new[] {torch.TensorIndex.Single(i)};
                var prob = probs[tensorIndices];

                return torch.multinomial(prob, 1);
            }).ToList();
            var final = torch.vstack(arr);

            return KeepDims ? final : final.squeeze(-1);
        }
    }
}