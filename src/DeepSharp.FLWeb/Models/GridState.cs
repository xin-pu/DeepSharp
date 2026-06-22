namespace DeepSharp.FLWeb.Models;

public class GridState
{
    public GridCell[] Cells { get; set; } = Array.Empty<GridCell>();
    public int PlayerIndex { get; set; }

    /// <summary>
    ///     Action index: 0=Up, 1=Down, 2=Left, 3=Right.
    /// </summary>
    public int Action { get; set; }

    public string ActionName { get; set; } = "";
    public float Reward { get; set; }
    public bool IsComplete { get; set; }
}
