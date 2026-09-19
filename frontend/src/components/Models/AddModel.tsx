import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import { z } from "zod"

import {
  ModelsService,
} from "@/client"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { LoadingButton } from "@/components/ui/loading-button"
import { Checkbox } from "@/components/ui/checkbox"
import useCustomToast from "@/hooks/useCustomToast"
import { useQueryClient } from "@tanstack/react-query"

const formSchema = z.object({
  name: z.string().min(1, { message: "Name is required" }),
  path: z.string().min(1, { message: "Path is required" }),
  size_bytes: z.coerce.number().min(0, { message: "Size must be positive" }),
  architecture: z.string().min(1, { message: "Architecture is required" }),
  parameter_count: z.coerce.number().optional(),
  quantization: z.string().min(1, { message: "Quantization is required" }),
  supports_embeddings: z.boolean().default(false),
  supports_vision: z.boolean().default(false),
  context_length: z.coerce.number().min(1, { message: "Context length is required" }),
  license: z.string().optional(),
  tags: z.array(z.string()).default([]),
  description: z.string().optional(),
  source: z.string().min(1, { message: "Source is required" }),
  source_repo_id: z.string().optional(),
  source_url: z.string().url().optional().or(z.literal("")),
  source_file: z.string().optional(),
})

type FormData = z.infer<typeof formSchema>

interface AddModelProps {
  isOpen: boolean
  onClose: () => void
}

export const AddModel = ({ isOpen, onClose }: AddModelProps) => {
  const queryClient = useQueryClient()
  const { showSuccessToast } = useCustomToast()

  const form = useForm<FormData>({
    resolver: zodResolver(formSchema),
    mode: "onBlur",
    defaultValues: {
      name: "",
      path: "",
      size_bytes: 0,
      architecture: "",
      parameter_count: undefined,
      quantization: "",
      supports_embeddings: false,
      supports_vision: false,
      context_length: 4096,
      license: "",
      tags: [],
      description: "",
      source: "huggingface",
      source_repo_id: "",
      source_url: "",
      source_file: "",
    },
  })

  const onSubmit = async (data: FormData) => {
    try {
      await ModelsService.createModel({ body: data })
      showSuccessToast("Model created successfully")
      onClose()
      form.reset()
      queryClient.invalidateQueries({ queryKey: ["models"] })
    } catch (error: any) {
      console.error(error)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Add New Model</DialogTitle>
          <DialogDescription>
            Register a new GGUF model for inference
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="name"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Model Name</FormLabel>
                    <FormControl>
                      <Input placeholder="llama-2-7b-chat.Q4_K_M.gguf" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="architecture"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Architecture</FormLabel>
                    <FormControl>
                      <Input placeholder="llama" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <FormField
              control={form.control}
              name="path"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>File Path</FormLabel>
                  <FormControl>
                    <Input placeholder="/models/llama-2-7b-chat.Q4_K_M.gguf" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="grid grid-cols-3 gap-4">
              <FormField
                control={form.control}
                name="size_bytes"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Size (bytes)</FormLabel>
                    <FormControl>
                      <Input type="number" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="parameter_count"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Parameters</FormLabel>
                    <FormControl>
                      <Input type="number" placeholder="7000000000" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="context_length"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Context Length</FormLabel>
                    <FormControl>
                      <Input type="number" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <FormField
              control={form.control}
              name="quantization"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Quantization</FormLabel>
                  <FormControl>
                    <Input placeholder="Q4_K_M" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="supports_embeddings"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-center space-x-3 space-y-0">
                    <FormControl>
                      <Checkbox
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <FormLabel className="font-normal">
                      Supports Embeddings
                    </FormLabel>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="supports_vision"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-center space-x-3 space-y-0">
                    <FormControl>
                      <Checkbox
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <FormLabel className="font-normal">
                      Supports Vision
                    </FormLabel>
                  </FormItem>
                )}
              />
            </div>

            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description</FormLabel>
                  <FormControl>
                    <textarea
                      placeholder="Model description..."
                      className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="border-t pt-4">
              <h4 className="text-sm font-medium mb-3">Source Information</h4>
              
              <FormField
                control={form.control}
                name="source"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Source</FormLabel>
                    <FormControl>
                      <Input placeholder="huggingface" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <div className="grid grid-cols-2 gap-4 mt-4">
                <FormField
                  control={form.control}
                  name="source_repo_id"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Repository ID</FormLabel>
                      <FormControl>
                        <Input placeholder="TheBloke/Llama-2-7B-Chat-GGUF" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="source_file"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Source File</FormLabel>
                      <FormControl>
                        <Input placeholder="llama-2-7b-chat.Q4_K_M.gguf" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <FormField
                control={form.control}
                name="source_url"
                render={({ field }) => (
                  <FormItem className="mt-4">
                    <FormLabel>Source URL</FormLabel>
                    <FormControl>
                      <Input
                        type="url"
                        placeholder="https://huggingface.co/..."
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <div className="flex justify-end gap-2 pt-4">
              <LoadingButton type="button" variant="outline" onClick={onClose}>
                Cancel
              </LoadingButton>
              <LoadingButton type="submit">
                Create Model
              </LoadingButton>
            </div>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}
