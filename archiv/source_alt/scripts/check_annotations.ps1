$base = 'C:\Users\s4126718\Documents\Kettenwirken\2026-02-26'
foreach ($v in @('Versuch_1','Versuch_2','Versuch_3','Versuch_4','Versuch_5','Versuch_6','Versuch_8','Versuch_9','Versuch_10','Versuch_11')) {
    $f = Join-Path $base "$v\annotations.json"
    if (Test-Path $f) {
        $content = Get-Content $f -Raw
        $j = ConvertFrom-Json $content
        $keys = @($j.PSObject.Properties.Name)
        Write-Host ("{0}: {1} Annotationen" -f $v, $keys.Count)
    } else {
        Write-Host ("{0}: keine annotations.json" -f $v)
    }
}
