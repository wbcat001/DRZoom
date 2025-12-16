# HDBSCAN階層クラスタリング可視化システム - レイアウト設計書

## レイアウト概要

`app_labeled.py`のUIレイアウトは、Bootstrap Grid System（12列レイアウト）を基盤とした5エリア構成で設計されています。各エリアが連動して、HDBSCANクラスタリング結果の包括的な分析環境を提供します。

## レイアウト構造図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Main Container (fluid=True)                       │
│ ┌───────────┬─────────────────────┬─────────────────────┬─────────────────┐ │
│ │     A     │          B          │          C          │        D        │ │
│ │  Control  │    DR Scatter       │    Dendrogram       │    Details      │ │
│ │  Panel    │    Visualization    │     Hierarchy       │    & Info       │ │
│ │ (Col: 2)  │     (Col: 4)        │     (Col: 4)        │   (Col: 2)      │ │
│ │           │                     │                     │                 │ │
│ └───────────┴─────────────────────┴─────────────────────┴─────────────────┘ │
│ ┌─────────────────────────────────────┬─────────────────────────────────────┐ │
│ │                E1                   │                E2                   │ │
│ │       Cluster Heatmap               │       Cluster Details               │ │
│ │         (Col: 6)                    │         (Col: 6)                    │ │
│ │                                     │                                     │ │
│ └─────────────────────────────────────┴─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## CSS高さ管理システム

### ビューポート全体使用の設定
```css
html, body, #react-entry-point {
    height: 100%;
    margin: 0;
}
.dbc-container-fluid {
    height: 100%;
}
```

### 高さ継承チェーン
```
Viewport (100vh) 
    → Container (h-100) 
        → Row (h-100, g-0) 
            → Col (h-100, p-2) 
                → Card (calc(100vh - 40px)) 
                    → CardBody (d-flex flex-column h-100)
```

## エリア別詳細設計

### A: Control Panel (width=2)

**目的**: パラメータ制御とアプリケーション設定

**構成要素:**
```python
dbc.Col([
    dbc.Card([
        dbc.CardBody([
            html.H4("Control Panel"),
            
            # データセット選択
            dcc.Dropdown(id='dataset-selector'),
            
            # DR手法選択  
            dbc.RadioItems(id='dr-method-selector'),
            
            # パラメータ設定（動的生成）
            html.Div(id='parameter-settings', className="flex-grow-1 overflow-auto"),
            
            # 実行ボタン
            dbc.Button(id='execute-button', className="mt-auto")
        ])
    ], className="h-100 d-flex flex-column")
], width=2, className="p-2 h-100")
```

**レイアウト特徴:**
- **縦方向Flexbox**: `d-flex flex-column`でコンテンツを縦配置
- **自動拡張**: `flex-grow-1`でパラメータエリアが残りスペースを占有
- **固定配置**: `mt-auto`で実行ボタンを下部に固定

### B: DR Visualization (width=4)

**目的**: UMAP/t-SNE/PCA埋め込み空間の2D散布図表示

**構成要素:**
```python
dbc.Col([
    dbc.Card([
        dbc.CardBody([
            html.H4("DR Visualization"),
            
            # インタラクション制御
            dbc.Row([
                dbc.Col([
                    dbc.RadioItems(id='dr-interaction-mode-toggle', inline=True)
                ], width=8),
                dbc.Col([
                    dbc.Checklist(id='dr-label-annotation-toggle')
                ], width=4)
            ]),
            
            # メインプロット
            dcc.Graph(id='dr-visualization-plot', className="flex-grow-1")
        ], className="d-flex flex-column p-3 h-100")
    ], style={'height': 'calc(100vh - 40px)'})
], width=4, className="p-2 h-100")
```

**インタラクション機能:**
- **Brush Selection**: ラッソ/ボックス選択によるマルチポイント選択
- **Zoom/Pan**: 詳細領域の拡大・移動操作
- **Label Annotation**: データポイントのラベル表示切り替え

### C: Dendrogram Hierarchy (width=4)

**目的**: 階層クラスタ構造のデンドログラム可視化

**構成要素:**
```python
dbc.Col([
    dbc.Card([
        dbc.CardBody([
            html.H4("Cluster Dendrogram"),
            
            # 3列制御パネル
            dbc.Row([
                dbc.Col([
                    dbc.RadioItems(id='dendro-interaction-mode-toggle', inline=True)
                ], width=4),
                dbc.Col([
                    dbc.Checklist(id='dendro-width-option-toggle')
                ], width=4),
                dbc.Col([
                    dbc.Checklist(id='dendro-label-annotation-toggle')  
                ], width=4)
            ]),
            
            # デンドログラムプロット
            dcc.Graph(id='dendrogram-plot', className="flex-grow-1")
        ], className="d-flex flex-column p-3 h-100")
    ], style={'height': 'calc(100vh - 40px)'})
], width=4, className="p-2 h-100")
```

**表示オプション:**
- **Node Selection**: ノードクリック・ホバーによる階層探索
- **Proportional Width**: クラスターサイズに応じた線幅調整
- **Label Annotation**: クラスター代表ラベル表示

### D: Details & Info Panel (width=2)

**目的**: 選択要素の詳細情報とシステム状態表示

**タブ構成:**
```python
dbc.Tabs(
    id='detail-info-tabs',
    children=[
        dbc.Tab(label='Point Details', tab_id='tab-point-details'),
        dbc.Tab(label='Selection Stats', tab_id='tab-selection-stats'),
        dbc.Tab(label='Cluster Size Dist', tab_id='tab-cluster-size'),
        dbc.Tab(label='System Log', tab_id='tab-system-log')
    ]
)
```

**表示情報:**
- **Point Details**: クリック点の詳細統計（ID、座標、ラベル、近傍情報）
- **Selection Stats**: 選択領域の統計サマリ（点数、特徴量統計）
- **Cluster Size Distribution**: クラスターサイズ分布の可視化
- **System Log**: アプリケーション操作履歴

### E1: Cluster Heatmap (width=6) 

**目的**: クラスター間類似度の行列表示

**機能:**
- **類似度指標**: KL距離、Bhattacharyya係数、Mahalanobis距離
- **Colorscale制御**: 正規/逆転カラースケール切り替え
- **Cluster Reorder**: 類似度に基づく階層的並び替え
- **選択連動**: 他ビューの選択に応じたハイライト表示

### E2: Cluster Details (width=6)

**目的**: 選択クラスターの詳細分析情報

**表示内容:**
- クラスターサイズと構成点数
- 代表単語とキーワードリスト
- Stabilityスコアと品質指標
- 類似クラスターランキング

## データストア設計

### 選択状態管理
```python
dcc.Store(id='selected-ids-store', data={
    'dr_selected_clusters': [],         # DRビューで選択されたクラスタ
    'dr_selected_points': [],           # DRビューで選択されたポイント
    'heatmap_clicked_clusters': [],     # ヒートマップでクリックされたクラスタ
    'heatmap_highlight_points': [],     # ヒートマップクリックによるハイライトポイント
    'dendrogram_clicked_clusters': [],  # デンドログラムでクリックされたクラスタ
    'dendrogram_highlight_points': [],  # デンドログラムクリックによるハイライトポイント
    'last_interaction_type': None       # インタラクション種別追跡
})
```

### ズーム状態保持
```python
dcc.Store(id='dr-zoom-store', data={
    'xaxis_range': None,
    'yaxis_range': None
})
```

## 動的パラメータ設定システム

### UMAP パラメータUI
```python
html.Div([
    html.Label("n_neighbors:"),
    dcc.Slider(
        id='umap-n-neighbors',
        min=5, max=50, step=1, value=15,
        marks={i: str(i) for i in range(5, 51, 10)},
        tooltip={'placement': 'bottom', 'always_visible': True}
    ),
    html.Label("min_dist:"),
    dcc.Slider(
        id='umap-min-dist', 
        min=0.0, max=0.99, step=0.01, value=0.1,
        marks={i/10: f"{i/10:.1f}" for i in range(0, 10, 2)},
        tooltip={'placement': 'bottom', 'always_visible': True}
    )
])
```

### t-SNE パラメータUI
```python
html.Div([
    html.Label("perplexity:"),
    dcc.Slider(
        id='tsne-perplexity',
        min=5, max=50, step=1, value=30,
        marks={i: str(i) for i in range(5, 51, 10)},
        tooltip={'placement': 'bottom', 'always_visible': True}
    )
])
```

### PCA パラメータUI
```python
html.Div([
    html.Label("n_components:"),
    dcc.Dropdown(
        id='pca-n-components',
        options=[
            {'label': '2', 'value': 2},
            {'label': '3', 'value': 3}
        ],
        value=2
    )
])
```

## レスポンシブデザイン戦略

### ブレークポイント対応
- **Bootstrap Grid**: 自動的なカラム調整
- **Fluid Container**: 画面幅に応じた伸縮
- **Flexbox Layout**: コンテンツの動的配置

### 画面サイズ最適化
```python
# 大画面用（デスクトップ）
style={'height': 'calc(100vh - 40px)'}  # ビューポート最大活用

# 中画面用（タブレット）  
className="d-flex flex-column"  # 縦積みレイアウト

# 小画面用（モバイル）
className="overflow-auto"  # スクロール対応
```

## パフォーマンス考慮事項

### レンダリング最適化
1. **グラフサイズ固定**: `calc(100vh - 40px)`で再描画最小化
2. **Flexbox活用**: CSS Gridより軽量なレイアウト
3. **オーバーフロー制御**: `overflow-auto`で部分スクロール

### メモリ効率化
1. **Store分離**: 機能別データストアで更新範囲限定
2. **条件付きレンダリング**: 表示状態に応じたコンポーネント生成
3. **遅延読み込み**: 大型データの段階的ロード

## アクセシビリティ配慮

### キーボード操作
- **Tab順序**: 論理的なフォーカス遷移
- **ARIA属性**: スクリーンリーダー対応
- **ショートカット**: 主要操作のキーボード割り当て

### 視覚的配慮
- **高コントラスト**: 色覚異常への対応
- **フォントサイズ**: 可読性確保
- **色以外の区別**: 形状・パターンでの情報伝達

## 拡張性設計

### モジュラー構造
```python
# エリア別コンポーネント関数化例
def create_control_panel():
    return dbc.Col([...], width=2)

def create_dr_visualization():  
    return dbc.Col([...], width=4)

def create_dendrogram_area():
    return dbc.Col([...], width=4)

def create_details_panel():
    return dbc.Col([...], width=2)
```

### 設定可能パラメータ
- **カラム幅**: Grid System比率調整
- **高さ設定**: ビューポート使用率変更
- **レイアウトモード**: 縦積み/横並び切り替え

## ブラウザ互換性

### 対応ブラウザ
- Chrome 80+（推奨）
- Firefox 75+
- Safari 13+  
- Edge 80+

### 非対応時の代替表示
```python
else:
    app.layout = html.Div([
        html.H3("Dash Bootstrap Components not available. Please install: pip install dash-bootstrap-components")
    ])
```

## D3.js実装のための詳細レイアウト仕様

### HTML構造設計

#### ルートコンテナ構造
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HDBSCAN Cluster Explorer</title>
    <style>
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .main-container {
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
    </style>
</head>
<body>
    <div class="main-container">
        <!-- 上段: 4パネルレイアウト -->
        <div class="top-row">
            <div class="control-panel"></div>
            <div class="dr-visualization"></div>
            <div class="dendrogram-area"></div>
            <div class="details-panel"></div>
        </div>
        
        <!-- 下段: 2パネルレイアウト -->
        <div class="bottom-row">
            <div class="cluster-heatmap"></div>
            <div class="cluster-details"></div>
        </div>
    </div>
</body>
</html>
```

#### CSS Grid レイアウト実装
```css
.main-container {
    display: grid;
    grid-template-rows: 1fr 400px; /* 上段: 可変, 下段: 400px固定 */
    height: 100vh;
    gap: 8px;
    padding: 8px;
}

.top-row {
    display: grid;
    grid-template-columns: 2fr 4fr 4fr 2fr; /* A:2, B:4, C:4, D:2 */
    gap: 8px;
    min-height: 0; /* グリッドアイテムの縮小を許可 */
}

.bottom-row {
    display: grid;
    grid-template-columns: 1fr 1fr; /* E1:6, E2:6 → 1:1 */
    gap: 8px;
    height: 400px;
}

/* 各パネルのベーススタイル */
.panel {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.panel-header {
    background: #f8f9fa;
    border-bottom: 1px solid #dee2e6;
    padding: 12px 16px;
    font-weight: 600;
    font-size: 16px;
    text-align: center;
}

.panel-content {
    flex: 1;
    padding: 16px;
    overflow: auto;
}
```

### エリア別D3.js実装仕様

#### A: Control Panel (width: 16.67%)
```javascript
// Control Panel D3.js実装
class ControlPanel {
    constructor(container) {
        this.container = d3.select(container);
        this.setupLayout();
    }
    
    setupLayout() {
        const panel = this.container
            .append('div')
            .attr('class', 'panel control-panel');
        
        // Header
        panel.append('div')
            .attr('class', 'panel-header')
            .text('Control Panel');
        
        const content = panel.append('div')
            .attr('class', 'panel-content')
            .style('display', 'flex')
            .style('flex-direction', 'column');
        
        // Dataset Selector
        this.createDatasetSelector(content);
        
        // DR Method Selector
        this.createDRMethodSelector(content);
        
        // Dynamic Parameters
        this.parameterContainer = content.append('div')
            .attr('class', 'parameter-container')
            .style('flex', '1')
            .style('overflow-y', 'auto');
        
        // Execute Button
        content.append('button')
            .attr('class', 'btn btn-primary execute-btn')
            .style('margin-top', 'auto')
            .style('padding', '12px')
            .style('border', 'none')
            .style('border-radius', '4px')
            .style('background', '#007bff')
            .style('color', 'white')
            .style('cursor', 'pointer')
            .text('Run Analysis')
            .on('click', () => this.executeAnalysis());
    }
    
    createDatasetSelector(parent) {
        const group = parent.append('div').attr('class', 'form-group');
        group.append('label').text('Dataset:');
        
        const select = group.append('select')
            .attr('class', 'form-control')
            .on('change', (event) => this.onDatasetChange(event.target.value));
        
        select.selectAll('option')
            .data([
                {label: 'Iris', value: 'iris'},
                {label: 'Digits', value: 'digits'},
                {label: 'Wine', value: 'wine'}
            ])
            .enter()
            .append('option')
            .attr('value', d => d.value)
            .text(d => d.label);
    }
    
    createDRMethodSelector(parent) {
        const group = parent.append('div').attr('class', 'form-group');
        group.append('label').text('DR Method:');
        
        const methods = ['UMAP', 'TSNE', 'PCA'];
        const radioGroup = group.append('div').attr('class', 'radio-group');
        
        methods.forEach(method => {
            const label = radioGroup.append('label')
                .style('display', 'block')
                .style('margin', '4px 0');
            
            label.append('input')
                .attr('type', 'radio')
                .attr('name', 'dr-method')
                .attr('value', method)
                .property('checked', method === 'UMAP')
                .on('change', () => this.onDRMethodChange(method));
            
            label.append('span')
                .style('margin-left', '8px')
                .text(method);
        });
    }
}
```

#### B: DR Visualization (width: 33.33%)
```javascript
// DR Scatter Plot D3.js実装
class DRVisualization {
    constructor(container) {
        this.container = d3.select(container);
        this.margin = {top: 20, right: 20, bottom: 40, left: 40};
        this.setupLayout();
        this.setupInteractions();
    }
    
    setupLayout() {
        const panel = this.container
            .append('div')
            .attr('class', 'panel dr-panel');
        
        // Header with controls
        const header = panel.append('div')
            .attr('class', 'panel-header')
            .style('display', 'flex')
            .style('justify-content', 'space-between')
            .style('align-items', 'center');
        
        header.append('h4').text('DR Visualization');
        
        const controls = header.append('div')
            .attr('class', 'interaction-controls');
        
        // Interaction mode toggle
        const modeToggle = controls.append('div')
            .attr('class', 'mode-toggle');
        
        ['brush', 'zoom'].forEach(mode => {
            const label = modeToggle.append('label')
                .style('margin-right', '12px');
            
            label.append('input')
                .attr('type', 'radio')
                .attr('name', 'interaction-mode')
                .attr('value', mode)
                .property('checked', mode === 'zoom')
                .on('change', () => this.setInteractionMode(mode));
            
            label.append('span')
                .style('margin-left', '4px')
                .text(mode === 'brush' ? 'Brush Selection' : 'Zoom/Pan');
        });
        
        // SVG container
        const content = panel.append('div')
            .attr('class', 'panel-content')
            .style('position', 'relative');
        
        this.svg = content.append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .style('display', 'block');
        
        // Responsive resize
        this.updateDimensions();
        window.addEventListener('resize', () => this.updateDimensions());
    }
    
    setupInteractions() {
        // Zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
                this.currentTransform = event.transform;
            });
        
        // Brush behavior
        this.brush = d3.brush()
            .on('start brush end', (event) => this.onBrush(event));
        
        this.interactionMode = 'zoom';
        this.setInteractionMode('zoom');
    }
    
    setInteractionMode(mode) {
        this.interactionMode = mode;
        
        if (mode === 'zoom') {
            this.svg.call(this.zoom);
            this.svg.select('.brush-layer').remove();
        } else {
            this.svg.on('.zoom', null);
            const brushLayer = this.svg.append('g')
                .attr('class', 'brush-layer')
                .call(this.brush);
        }
    }
    
    updateDimensions() {
        const rect = this.container.node().getBoundingClientRect();
        this.width = rect.width - this.margin.left - this.margin.right;
        this.height = rect.height - this.margin.top - this.margin.bottom;
        
        this.svg
            .attr('width', rect.width)
            .attr('height', rect.height);
        
        if (this.g) {
            this.g.attr('transform', `translate(${this.margin.left},${this.margin.top})`);
        }
    }
}
```

#### C: Dendrogram (width: 33.33%)
```javascript
// Dendrogram D3.js実装  
class DendrogramVisualization {
    constructor(container) {
        this.container = d3.select(container);
        this.margin = {top: 40, right: 20, bottom: 40, left: 40};
        this.setupLayout();
    }
    
    setupLayout() {
        const panel = this.container
            .append('div')
            .attr('class', 'panel dendrogram-panel');
        
        // Header with controls
        const header = panel.append('div')
            .attr('class', 'panel-header')
            .style('display', 'flex')
            .style('flex-direction', 'column')
            .style('gap', '8px');
        
        header.append('h4')
            .style('margin', '0')
            .style('text-align', 'center')
            .text('Cluster Dendrogram');
        
        // Control row with 3 columns
        const controlRow = header.append('div')
            .style('display', 'grid')
            .style('grid-template-columns', '1fr 1fr 1fr')
            .style('gap', '8px')
            .style('font-size', '12px');
        
        // Interaction mode
        const interactionCol = controlRow.append('div');
        interactionCol.append('label')
            .append('input')
            .attr('type', 'radio')
            .attr('name', 'dendro-mode')
            .attr('value', 'node')
            .property('checked', true);
        interactionCol.append('span').text(' Node Selection');
        
        // Proportional width
        const widthCol = controlRow.append('div');
        widthCol.append('label')
            .append('input')
            .attr('type', 'checkbox')
            .attr('id', 'prop-width')
            .on('change', (event) => this.toggleProportionalWidth(event.target.checked));
        widthCol.append('span').text(' Proportional Width');
        
        // Show labels
        const labelCol = controlRow.append('div');
        labelCol.append('label')
            .append('input')
            .attr('type', 'checkbox')
            .attr('id', 'show-labels')
            .on('change', (event) => this.toggleLabels(event.target.checked));
        labelCol.append('span').text(' Show Labels');
        
        // SVG container
        const content = panel.append('div')
            .attr('class', 'panel-content');
        
        this.svg = content.append('svg')
            .attr('width', '100%')
            .attr('height', '100%');
        
        this.g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
        
        this.updateDimensions();
    }
    
    renderDendrogram(data) {
        // Clear previous content
        this.g.selectAll('*').remove();
        
        // Create scales
        const xScale = d3.scaleLinear()
            .domain([0, data.maxX])
            .range([0, this.width]);
        
        const yScale = d3.scaleLinear()
            .domain([0, data.maxY])
            .range([this.height, 0]);
        
        // Draw connections
        this.g.selectAll('.dendrogram-link')
            .data(data.links)
            .enter()
            .append('path')
            .attr('class', 'dendrogram-link')
            .attr('d', d => this.createLinkPath(d, xScale, yScale))
            .style('fill', 'none')
            .style('stroke', '#333')
            .style('stroke-width', d => this.proportionalWidth ? d.weight || 1 : 1)
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip());
        
        // Draw nodes
        this.g.selectAll('.dendrogram-node')
            .data(data.nodes)
            .enter()
            .append('circle')
            .attr('class', 'dendrogram-node')
            .attr('cx', d => xScale(d.x))
            .attr('cy', d => yScale(d.y))
            .attr('r', 3)
            .style('fill', d => d.highlighted ? '#ff0000' : '#333')
            .on('click', (event, d) => this.onNodeClick(d));
    }
}
```

#### D: Details Panel (width: 16.67%)
```javascript
// Details Panel D3.js実装
class DetailsPanel {
    constructor(container) {
        this.container = d3.select(container);
        this.activeTab = 'point-details';
        this.setupLayout();
    }
    
    setupLayout() {
        const panel = this.container
            .append('div')
            .attr('class', 'panel details-panel');
        
        panel.append('div')
            .attr('class', 'panel-header')
            .text('Detail & Info');
        
        const content = panel.append('div')
            .attr('class', 'panel-content')
            .style('display', 'flex')
            .style('flex-direction', 'column');
        
        // Tab navigation
        const tabNav = content.append('div')
            .attr('class', 'tab-nav')
            .style('display', 'flex')
            .style('border-bottom', '1px solid #dee2e6')
            .style('margin-bottom', '12px');
        
        const tabs = [
            {id: 'point-details', label: 'Point Details'},
            {id: 'selection-stats', label: 'Selection Stats'},
            {id: 'cluster-size', label: 'Cluster Size'},
            {id: 'system-log', label: 'System Log'}
        ];
        
        tabs.forEach(tab => {
            tabNav.append('button')
                .attr('class', 'tab-button')
                .style('padding', '8px 12px')
                .style('border', 'none')
                .style('background', tab.id === this.activeTab ? '#007bff' : '#f8f9fa')
                .style('color', tab.id === this.activeTab ? 'white' : '#333')
                .style('cursor', 'pointer')
                .style('font-size', '12px')
                .text(tab.label)
                .on('click', () => this.switchTab(tab.id));
        });
        
        // Tab content
        this.tabContent = content.append('div')
            .attr('class', 'tab-content')
            .style('flex', '1')
            .style('overflow-y', 'auto');
        
        this.updateTabContent();
    }
    
    switchTab(tabId) {
        this.activeTab = tabId;
        
        // Update tab buttons
        this.container.selectAll('.tab-button')
            .style('background', (d, i) => {
                const tabs = ['point-details', 'selection-stats', 'cluster-size', 'system-log'];
                return tabs[i] === tabId ? '#007bff' : '#f8f9fa';
            })
            .style('color', (d, i) => {
                const tabs = ['point-details', 'selection-stats', 'cluster-size', 'system-log'];
                return tabs[i] === tabId ? 'white' : '#333';
            });
        
        this.updateTabContent();
    }
}
```

#### E1: Cluster Heatmap (width: 50%)
```javascript
// Heatmap D3.js実装
class ClusterHeatmap {
    constructor(container) {
        this.container = d3.select(container);
        this.margin = {top: 60, right: 60, bottom: 60, left: 60};
        this.setupLayout();
    }
    
    setupLayout() {
        const panel = this.container
            .append('div')
            .attr('class', 'panel heatmap-panel');
        
        // Header with controls
        const header = panel.append('div')
            .attr('class', 'panel-header')
            .style('display', 'flex')
            .style('justify-content', 'space-between')
            .style('align-items', 'center');
        
        header.append('h4').text('Cluster Similarity');
        
        const controls = header.append('div')
            .style('display', 'flex')
            .style('gap', '12px');
        
        // Similarity metric selector
        const metricSelect = controls.append('select')
            .attr('class', 'form-control')
            .style('padding', '4px')
            .on('change', (event) => this.changeSimilarityMetric(event.target.value));
        
        ['KL Divergence', 'Bhattacharyya', 'Mahalanobis'].forEach(metric => {
            metricSelect.append('option')
                .attr('value', metric.toLowerCase().replace(' ', '_'))
                .text(metric);
        });
        
        // Reverse colorscale toggle
        const reverseLabel = controls.append('label')
            .style('font-size', '12px');
        reverseLabel.append('input')
            .attr('type', 'checkbox')
            .on('change', (event) => this.toggleReverseColorscale(event.target.checked));
        reverseLabel.append('span').text(' Reverse');
        
        // SVG container
        const content = panel.append('div')
            .attr('class', 'panel-content');
        
        this.svg = content.append('svg')
            .attr('width', '100%')
            .attr('height', '100%');
        
        this.g = this.svg.append('g');
        
        this.updateDimensions();
    }
    
    renderHeatmap(data) {
        const cellSize = Math.min(
            this.width / data.clusters.length,
            this.height / data.clusters.length
        );
        
        // Create color scale
        const colorScale = d3.scaleSequential(d3.interpolateBlues)
            .domain(d3.extent(data.similarities.flat()));
        
        // Draw cells
        this.g.selectAll('.heatmap-cell')
            .data(data.similarities.flat())
            .enter()
            .append('rect')
            .attr('class', 'heatmap-cell')
            .attr('x', (d, i) => (i % data.clusters.length) * cellSize)
            .attr('y', (d, i) => Math.floor(i / data.clusters.length) * cellSize)
            .attr('width', cellSize)
            .attr('height', cellSize)
            .style('fill', d => colorScale(d))
            .style('stroke', 'white')
            .style('stroke-width', 1)
            .on('click', (event, d) => this.onCellClick(event, d))
            .on('mouseover', (event, d) => this.showTooltip(event, d));
    }
}
```

### データバインディングとイベント管理

#### 状態管理システム
```javascript
// Global State Manager
class StateManager {
    constructor() {
        this.state = {
            selectedClusters: [],
            selectedPoints: [],
            heatmapClickedClusters: [],
            dendrogramClickedClusters: [],
            lastInteractionType: null,
            drZoomState: { x: [0, 1], y: [0, 1] },
            parameters: {
                drMethod: 'UMAP',
                umapParams: { n_neighbors: 15, min_dist: 0.1 },
                tsneParams: { perplexity: 30 },
                pcaParams: { n_components: 2 }
            }
        };
        
        this.listeners = new Map();
    }
    
    subscribe(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }
    
    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => callback(data));
        }
    }
    
    updateState(updates) {
        Object.assign(this.state, updates);
        this.emit('stateChanged', this.state);
    }
}

// Global instance
const stateManager = new StateManager();
```

#### 色分けシステム
```javascript
// Color Highlight System
const HIGHLIGHT_COLORS = {
    default: '#4A90E2',
    defaultDimmed: '#B8D4F0',
    drSelection: '#FFA500',
    heatmapClick: '#FF0000',
    heatmapToDr: '#FF1493',
    dendrogramToDr: '#32CD32'
};

class ColorManager {
    static getPointColor(pointId, state) {
        // Priority order: heatmap > dr selection > dendrogram > default
        if (state.heatmapClickedClusters.includes(pointId)) {
            return HIGHLIGHT_COLORS.heatmapClick;
        }
        if (state.selectedPoints.includes(pointId)) {
            return HIGHLIGHT_COLORS.drSelection;
        }
        if (state.dendrogramClickedClusters.includes(pointId)) {
            return HIGHLIGHT_COLORS.dendrogramToDr;
        }
        
        // Dimmed if other selection exists
        const hasOtherSelection = state.selectedPoints.length > 0 || 
                                 state.heatmapClickedClusters.length > 0;
        return hasOtherSelection ? HIGHLIGHT_COLORS.defaultDimmed : HIGHLIGHT_COLORS.default;
    }
}
```

### レスポンシブ対応

#### メディアクエリ対応
```css
/* Tablet (768px - 1024px) */
@media screen and (max-width: 1024px) {
    .top-row {
        grid-template-columns: 1fr; /* 縦積みレイアウト */
        grid-template-rows: auto auto auto auto;
    }
    
    .bottom-row {
        grid-template-columns: 1fr; /* 縦積みレイアウト */
        grid-template-rows: 200px 200px;
        height: 400px;
    }
}

/* Mobile (< 768px) */
@media screen and (max-width: 768px) {
    .main-container {
        grid-template-rows: auto; /* 高さ自動調整 */
        height: auto;
        min-height: 100vh;
    }
    
    .panel {
        min-height: 300px; /* 最小高さ確保 */
    }
    
    .panel-header {
        font-size: 14px;
        padding: 8px 12px;
    }
}
```

#### 動的リサイズ対応
```javascript
// Responsive Resize Handler
class ResponsiveManager {
    constructor() {
        this.components = [];
        window.addEventListener('resize', () => this.handleResize());
    }
    
    register(component) {
        this.components.push(component);
    }
    
    handleResize() {
        // Debounce resize events
        clearTimeout(this.resizeTimeout);
        this.resizeTimeout = setTimeout(() => {
            this.components.forEach(component => {
                if (component.updateDimensions) {
                    component.updateDimensions();
                }
            });
        }, 150);
    }
}
```

このレイアウト設計により、複雑なHDBSCANクラスタリング結果を直感的に探索できる統合環境が実現されています。各エリアの独立性と連動性のバランスを取ることで、効率的なデータ分析ワークフローを支援します。
