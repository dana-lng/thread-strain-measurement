$base = 'C:\Users\s4126718\Documents\Kettenwirken\2026-02-26'
Get-ChildItem $base -Directory | ForEach-Object {
    $dir = $_
    $n = (Get-ChildItem $dir.FullName -Filter '*.tif' -ErrorAction SilentlyContinue).Count
    if ($n -gt 0) {
        Write-Host ("{0}: {1} Bilder" -f $dir.Name, $n)
    }
}
