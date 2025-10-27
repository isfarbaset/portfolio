"""
Generate PNG visualizations for Wicked Spotify analysis
"""

import plotly.graph_objects as go
import plotly.express as px
import json
import pandas as pd
from pathlib import Path

# Create output directory
output_dir = Path('/Users/isfarbaset/Documents/portfolio/wicked-visualizations')
output_dir.mkdir(exist_ok=True)

# Load data
with open('/Users/isfarbaset/Documents/wicked-tiktok-analysis/data/processed/viz_data.json', 'r') as f:
    viz_data = json.load(f)

df = pd.read_csv('/Users/isfarbaset/Documents/wicked-tiktok-analysis/data/processed/spotify_analysis_results.csv')

print("Generating visualizations...")

# 1. Top 10 Songs by Popularity
print("1. Top 10 Songs by Popularity...")
top_songs = viz_data['top_10']
songs = [s['clean_name'] for s in top_songs]
popularity = [s['popularity'] for s in top_songs]
duration = [s['duration_min'] for s in top_songs]

colors = ['#00A651' if i < 5 else '#E91E8C' for i in range(len(songs))]

fig1 = go.Figure(data=[
    go.Bar(
        x=songs,
        y=popularity,
        marker=dict(
            color=colors,
            line=dict(color='#1a1a1a', width=1)
        ),
        text=[f'{v}' for v in popularity],
        textposition='outside',
        textfont=dict(size=11, color='#1a1a1a', family='Segoe UI', weight='bold'),
        hovertemplate='<b>%{x}</b><br>Popularity: %{y}<br>Duration: %{customdata:.2f} min<extra></extra>',
        customdata=duration
    )
])

fig1.update_layout(
    title=dict(
        text='Top 10 Most Popular Wicked Tracks on Spotify',
        font=dict(size=18, color='#1a1a1a', family='Segoe UI', weight='bold')
    ),
    xaxis=dict(
        title='',
        tickangle=-45,
        tickfont=dict(size=10, family='Segoe UI'),
        showgrid=False
    ),
    yaxis=dict(
        title=dict(text='Spotify Popularity Score', font=dict(size=12)),
        showgrid=True,
        gridcolor='#f0f0f0',
        range=[0, 65]
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=450,
    margin=dict(b=140, t=60),
    font=dict(family='Segoe UI', size=12, color='#333')
)

fig1.write_image(str(output_dir / '1_top_10_popularity.png'), width=1000, height=450)

# 2. Duration vs Popularity Scatter
print("2. Duration vs Popularity Scatter...")
scatter_data = viz_data['scatter']
duration_list = [s['duration_min'] for s in scatter_data]
pop_list = [s['popularity'] for s in scatter_data]
names = [s['clean_name'] for s in scatter_data]

fig2 = go.Figure(data=go.Scatter(
    x=duration_list,
    y=pop_list,
    mode='markers',
    marker=dict(
        size=12,
        color=pop_list,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Popularity'),
        line=dict(width=1, color='#1a1a1a')
    ),
    text=names,
    hovertemplate='<b>%{text}</b><br>Duration: %{x:.2f} min<br>Popularity: %{y}<extra></extra>'
))

fig2.update_layout(
    title=dict(
        text='Duration vs. Popularity: The Surprising Non-Relationship',
        font=dict(size=18, color='#1a1a1a', family='Segoe UI', weight='bold')
    ),
    xaxis=dict(
        title='Duration (minutes)',
        showgrid=True,
        gridcolor='#f0f0f0'
    ),
    yaxis=dict(
        title='Spotify Popularity Score',
        showgrid=True,
        gridcolor='#f0f0f0'
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=500,
    font=dict(family='Segoe UI', size=12, color='#333')
)

fig2.write_image(str(output_dir / '2_duration_vs_popularity.png'), width=1000, height=500)

# 3. Cluster Analysis
print("3. Cluster Analysis...")

# Filter for clarity (remove 0-popularity tracks)
df_filtered = df[df['popularity'] > 0].copy()

# Clean track names
df_filtered['clean_name'] = df_filtered['track_name'].str.extract(r'^"?([^"]+?)"?\s*-?\s*From')[0]
df_filtered['clean_name'] = df_filtered['clean_name'].fillna(df_filtered['track_name'].str.split(' - ').str[0])

# Color by cluster
cluster_colors = {0: '#00A651', 1: '#8B5CF6', 2: '#E91E8C'}
df_filtered['color'] = df_filtered['cluster'].map(cluster_colors)

fig3 = go.Figure()

for cluster_id in sorted(df_filtered['cluster'].unique()):
    cluster_data = df_filtered[df_filtered['cluster'] == cluster_id]
    cluster_names = ['Epic Showstoppers', 'International', 'Mid-Tier']
    
    fig3.add_trace(go.Scatter(
        x=cluster_data['PC1'],
        y=cluster_data['PC2'],
        mode='markers',
        name=cluster_names[cluster_id],
        marker=dict(
            size=10,
            color=cluster_colors[cluster_id],
            line=dict(color='#1a1a1a', width=1),
            opacity=0.7
        ),
        text=cluster_data['clean_name'],
        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
    ))

fig3.update_layout(
    title=dict(
        text='Three Song Archetypes: K-Means Clustering (PCA Projection)',
        font=dict(size=18, color='#1a1a1a', family='Segoe UI', weight='bold')
    ),
    xaxis=dict(
        title='Principal Component 1',
        showgrid=True,
        gridcolor='#f0f0f0'
    ),
    yaxis=dict(
        title='Principal Component 2',
        showgrid=True,
        gridcolor='#f0f0f0'
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=500,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    ),
    font=dict(family='Segoe UI', size=12, color='#333')
)

fig3.write_image(str(output_dir / '3_cluster_analysis.png'), width=1000, height=500)

# 4. Cluster Statistics
print("4. Cluster Statistics...")
cluster_stats_data = viz_data['clusters']

cluster_names_list = ['Epic Showstoppers', 'International', 'Mid-Tier']

fig4 = go.Figure()

fig4.add_trace(go.Bar(
    name='Avg Popularity',
    x=cluster_names_list,
    y=[c['avg_popularity'] for c in cluster_stats_data],
    marker=dict(color='#00A651'),
    text=[f"{c['avg_popularity']:.1f}" for c in cluster_stats_data],
    textposition='outside'
))

fig4.add_trace(go.Bar(
    name='Avg Duration (min)',
    x=cluster_names_list,
    y=[c['avg_duration'] for c in cluster_stats_data],
    marker=dict(color='#E91E8C'),
    text=[f"{c['avg_duration']:.2f}" for c in cluster_stats_data],
    textposition='outside'
))

fig4.update_layout(
    title=dict(
        text='Cluster Characteristics: Popularity and Duration',
        font=dict(size=18, color='#1a1a1a', family='Segoe UI', weight='bold')
    ),
    xaxis=dict(title=''),
    yaxis=dict(title='Value'),
    barmode='group',
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=450,
    font=dict(family='Segoe UI', size=12, color='#333')
)

fig4.write_image(str(output_dir / '4_cluster_stats.png'), width=1000, height=450)

# 5. Popularity Distribution
print("5. Popularity Distribution...")
fig5 = go.Figure(data=[go.Histogram(
    x=df['popularity'],
    nbinsx=15,
    marker=dict(
        color='#00A651',
        line=dict(color='#1a1a1a', width=1)
    ),
    hovertemplate='Popularity Range: %{x}<br>Count: %{y}<extra></extra>'
)])

fig5.update_layout(
    title=dict(
        text='Distribution of Popularity Scores Across All Tracks',
        font=dict(size=18, color='#1a1a1a', family='Segoe UI', weight='bold')
    ),
    xaxis=dict(
        title='Spotify Popularity Score',
        showgrid=False
    ),
    yaxis=dict(
        title='Number of Tracks',
        showgrid=True,
        gridcolor='#f0f0f0'
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=450,
    font=dict(family='Segoe UI', size=12, color='#333')
)

fig5.write_image(str(output_dir / '5_popularity_distribution.png'), width=1000, height=450)

# 6. Duration by Popularity Tier (Violin Plot)
print("6. Duration by Popularity Tier...")
# Create popularity tiers
df['popularity_tier'] = pd.cut(df['popularity'], 
                               bins=[0, 35, 50, 100],
                               labels=['Low (0-35)', 'Medium (36-50)', 'High (51+)'])

fig6 = go.Figure()

for tier in ['Low (0-35)', 'Medium (36-50)', 'High (51+)']:
    tier_data = df[df['popularity_tier'] == tier]['duration_min']
    fig6.add_trace(go.Violin(
        y=tier_data,
        name=tier,
        box_visible=True,
        meanline_visible=True,
        fillcolor=['#E91E8C', '#8B5CF6', '#00A651'][['Low (0-35)', 'Medium (36-50)', 'High (51+)'].index(tier)],
        opacity=0.6,
        line=dict(color='#1a1a1a')
    ))

fig6.update_layout(
    title=dict(
        text='Song Duration Distribution Across Popularity Tiers',
        font=dict(size=18, color='#1a1a1a', family='Segoe UI', weight='bold')
    ),
    yaxis=dict(
        title='Duration (minutes)',
        showgrid=True,
        gridcolor='#f0f0f0'
    ),
    xaxis=dict(title='Popularity Tier'),
    plot_bgcolor='white',
    paper_bgcolor='white',
    height=500,
    showlegend=False,
    font=dict(family='Segoe UI', size=12, color='#333')
)

fig6.write_image(str(output_dir / '6_duration_by_tier.png'), width=1000, height=500)

print(f"\n✅ All visualizations saved to: {output_dir}")
print("\nGenerated files:")
for i in range(1, 7):
    files = list(output_dir.glob(f'{i}_*.png'))
    for f in files:
        print(f"  - {f.name}")
