import type { ReactNode } from "react"
import type { Model } from "@/client"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"

export type ModelValueKey = "id" | "path"

export function modelBasename(value?: string | null): string {
  if (!value) return ""
  const trimmed = value.replace(/\/+$/, "")
  return trimmed.split("/").pop() || trimmed
}

function OptionContent({
  title,
  subtitle,
}: {
  title: string
  subtitle?: string
}) {
  return (
    <span className="flex min-w-0 flex-1 flex-col text-left leading-tight">
      <span className="truncate font-medium">{title}</span>
      {subtitle ? (
        <span className="truncate text-xs font-normal text-muted-foreground">
          {subtitle}
        </span>
      ) : null}
    </span>
  )
}

export type ModelSelectExtraItem = {
  value: string
  label: ReactNode
  triggerText?: string
}

type ModelSelectProps = {
  models: Model[]
  value: string
  onValueChange: (value: string) => void
  modelType?: string
  valueKey?: ModelValueKey
  placeholder?: string
  extraItems?: ModelSelectExtraItem[]
  className?: string
  disabled?: boolean
  id?: string
}

export function ModelSelect({
  models,
  value,
  onValueChange,
  modelType,
  valueKey = "id",
  placeholder,
  extraItems,
  className,
  disabled,
  id,
}: ModelSelectProps) {
  const filtered = modelType
    ? models.filter((model) => model.model_type === modelType)
    : models

  const selected = filtered.find((model) => model[valueKey] === value)
  const selectedExtra = extraItems?.find((item) => item.value === value)
  const triggerLabel: ReactNode = selected
    ? modelBasename(selected.name || selected.path)
    : selectedExtra
      ? (selectedExtra.triggerText ?? selectedExtra.label)
      : valueKey === "path" && value
        ? modelBasename(value)
        : undefined

  return (
    <Select
      value={value || undefined}
      onValueChange={onValueChange}
      disabled={disabled}
    >
      <SelectTrigger id={id} className={cn("w-full min-w-0", className)}>
        <SelectValue
          className="min-w-0 flex-1 truncate text-left"
          placeholder={placeholder}
        >
          {triggerLabel}
        </SelectValue>
      </SelectTrigger>
      <SelectContent className="max-h-80">
        {extraItems?.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
        {filtered.map((model) => {
          const title = modelBasename(model.name || model.path)
          const itemValue =
            valueKey === "id" ? (model.id ?? model.name) : (model.path ?? "")
          return (
            <SelectItem
              key={`${itemValue}`}
              value={itemValue}
              textValue={title}
              className="items-center py-2"
            >
              <OptionContent title={title} subtitle={model.path} />
            </SelectItem>
          )
        })}
      </SelectContent>
    </Select>
  )
}

export function ModelPathFallbackItem({ path }: { path: string }) {
  return <OptionContent title={modelBasename(path)} subtitle={path} />
}

export default ModelSelect
