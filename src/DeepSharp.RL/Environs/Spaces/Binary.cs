namespace DeepSharp.RL.Environs.Spaces
{
    /// <summary>
    ///     It's Space only support [0,1]
    /// </summary>
    public class Binary : Disperse
    {
        public Binary(torch.ScalarType dtype = torch.ScalarType.Int32,
            DeviceType deviceType = DeviceType.CUDA, long seed = 1)
            : base(2, dtype, deviceType, seed)
        {
            N = 1;
        }
    }
}