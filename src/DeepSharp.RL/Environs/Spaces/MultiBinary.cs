namespace DeepSharp.RL.Environs.Spaces
{
    /// <summary>
    ///     Todo
    /// </summary>
    public class MultiBinary : MultiDisperse
    {
        public MultiBinary(torch.Tensor low, torch.Tensor high, long[] shape, torch.ScalarType type,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1) : base(low, high, shape, type, deviceType, seed)
        {
        }
    }
}