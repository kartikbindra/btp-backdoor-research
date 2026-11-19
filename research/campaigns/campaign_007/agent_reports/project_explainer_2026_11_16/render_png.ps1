# Deterministic raster exports of the simple SVG teaching diagrams using Windows GDI+.
# No GUI, network, model execution, or external dependencies.
Add-Type -AssemblyName System.Drawing
$ErrorActionPreference = 'Stop'
$diagramDir = Join-Path $PSScriptRoot 'diagrams'
$textOverflow = @()
function ColorOf([string]$value) { return [System.Drawing.ColorTranslator]::FromHtml($value) }
foreach ($sourceFile in Get-ChildItem -LiteralPath $diagramDir -Filter '*.svg') {
    [xml]$doc = Get-Content -LiteralPath $sourceFile.FullName -Encoding UTF8 -Raw
    $canvasWidth = [int]$doc.svg.width
    $canvasHeight = [int]$doc.svg.height
    $bitmap = New-Object System.Drawing.Bitmap($canvasWidth, $canvasHeight)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $format = [System.Drawing.StringFormat]::GenericTypographic.Clone()
    foreach ($el in $doc.svg.ChildNodes) {
        switch ($el.LocalName) {
            'rect' {
                $x = if ($el.HasAttribute('x')) { [float]$el.x } else { 0 }
                $y = if ($el.HasAttribute('y')) { [float]$el.y } else { 0 }
                $brush = New-Object System.Drawing.SolidBrush((ColorOf $el.fill))
                $rect = New-Object System.Drawing.RectangleF($x, $y, [float]$el.width, [float]$el.height)
                $graphics.FillRectangle($brush, $rect)
                if ($el.HasAttribute('stroke')) {
                    $pen = New-Object System.Drawing.Pen((ColorOf $el.stroke), 1)
                    $graphics.DrawRectangle($pen,$x,$y,[float]$el.width,[float]$el.height)
                    $pen.Dispose()
                }
                $brush.Dispose()
            }
            'circle' {
                $radius = [float]$el.r
                $brush = New-Object System.Drawing.SolidBrush((ColorOf $el.fill))
                $graphics.FillEllipse($brush,([float]$el.cx-$radius),([float]$el.cy-$radius),2*$radius,2*$radius)
                $brush.Dispose()
            }
            'path' {
                $coordinates = [regex]::Matches([string]$el.d,'-?\d+(?:\.\d+)?') | ForEach-Object { [float]$_.Value }
                if ($coordinates.Count -lt 3) { continue }
                $pen = New-Object System.Drawing.Pen((ColorOf $el.stroke), [float]$el.GetAttribute('stroke-width'))
                if ($el.HasAttribute('marker-end')) { $pen.EndCap = [System.Drawing.Drawing2D.LineCap]::ArrowAnchor }
                if ($coordinates.Count -eq 4) {
                    $graphics.DrawLine($pen,$coordinates[0],$coordinates[1],$coordinates[2],$coordinates[3])
                } elseif ($coordinates.Count -eq 3) {
                    $graphics.DrawLine($pen,$coordinates[0],$coordinates[1],$coordinates[0],$coordinates[2])
                }
                $pen.Dispose()
            }
            'text' {
                $style = if ($el.GetAttribute('font-weight') -eq '700') { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
                $font = New-Object System.Drawing.Font('Segoe UI',[float]$el.GetAttribute('font-size'),$style,[System.Drawing.GraphicsUnit]::Pixel)
                $brush = New-Object System.Drawing.SolidBrush((ColorOf $el.fill))
                $ascent = $font.Size * $font.FontFamily.GetCellAscent($style) / $font.FontFamily.GetEmHeight($style)
                $measure = $graphics.MeasureString($el.InnerText,$font,2000,$format)
                if ([float]$el.x + $measure.Width -gt $canvasWidth-10) {
                    $textOverflow += "$($sourceFile.Name): $($el.InnerText)"
                }
                $graphics.DrawString($el.InnerText,$font,$brush,[float]$el.x,([float]$el.y-$ascent),$format)
                $brush.Dispose()
                $font.Dispose()
            }
        }
    }
    $targetFile = [System.IO.Path]::ChangeExtension($sourceFile.FullName,'.png')
    $bitmap.Save($targetFile,[System.Drawing.Imaging.ImageFormat]::Png)
    $format.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
}
if ($textOverflow.Count) { throw ('Diagram text exceeds canvas: '+($textOverflow -join ' | ')) }
Write-Output 'Rendered 7 PNG exports. All diagram text fits the canvas.'
