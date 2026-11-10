{{/*
Expand the name of the chart.
*/}}
{{- define "cad-ingest.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "cad-ingest.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "cad-ingest.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cad-ingest.labels" -}}
helm.sh/chart: {{ include "cad-ingest.chart" . }}
app.kubernetes.io/name: {{ include "cad-ingest.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Values.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels of one component (api or worker), from a dict of context and component
*/}}
{{- define "cad-ingest.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cad-ingest.name" .context }}
app.kubernetes.io/instance: {{ .context.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "cad-ingest.image" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion }}
{{- if .Values.image.digest }}
{{- printf "%s:%s@%s" .Values.image.repository $tag .Values.image.digest }}
{{- else }}
{{- printf "%s:%s" .Values.image.repository $tag }}
{{- end }}
{{- end }}

{{/*
Environment both processes need
*/}}
{{- define "cad-ingest.commonEnv" -}}
- name: REDIS_URL
  {{- if .Values.redis.existingSecret }}
  valueFrom:
    secretKeyRef:
      name: {{ .Values.redis.existingSecret }}
      key: {{ .Values.redis.secretKey }}
  {{- else }}
  value: {{ required "Set redis.url or redis.existingSecret: cad-ingest keeps its queues and job state in Redis." .Values.redis.url | quote }}
  {{- end }}
- name: MAX_FILE_SIZE_BYTES
  value: {{ .Values.maxFileSizeBytes | int64 | quote }}
{{- end }}

{{/*
Worker probe: registered in Redis = alive. A restart would end the running conversions, so it tolerates a Redis
failover of several minutes.
*/}}
{{- define "cad-ingest.workerProbe" -}}
exec:
  command:
    - worker
    - check
initialDelaySeconds: 30
periodSeconds: 60
timeoutSeconds: 10
failureThreshold: 5
{{- end }}

{{/*
Pod settings both deployments share, from a dict of context and component; the worker also gets the profiles and
its additional volumes
*/}}
{{- define "cad-ingest.podSpec" -}}
{{- $values := .context.Values }}
{{- with $values.imagePullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 2 }}
{{- end }}
automountServiceAccountToken: false
{{- with $values.podSecurityContext }}
securityContext:
  {{- toYaml . | nindent 2 }}
{{- end }}
volumes:
  - name: tmp
    emptyDir: {}
{{- if eq .component "worker" }}
{{- if $values.worker.profiles }}
  - name: profiles
    configMap:
      name: {{ include "cad-ingest.fullname" .context }}-profiles
{{- end }}
{{- with $values.worker.extraVolumes }}
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}
{{- with $values.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with $values.affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with $values.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}
